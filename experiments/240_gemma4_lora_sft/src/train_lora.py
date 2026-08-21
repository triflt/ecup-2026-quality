"""SFT LoRA training for Gemma-4-E4B-it on QC task."""
import os
import sys
import json
import yaml
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Gemma4ForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
GEMMA_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/models/gemma-4-E4B-it'
SUBMIT_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/submit'
if SUBMIT_DIR not in sys.path:
    sys.path.insert(0, SUBMIT_DIR)

from src.ocr_stage import image_paths_for
from src.gemma_stage import IMG_MAX_EDGE
from PIL import Image


class QCChatDataset(Dataset):
    def __init__(self, jsonl_path: str, processor, max_length: int = 3072):
        self.rows = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        s = IMG_MAX_EDGE / max(w, h)
        if s < 1.0:
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return img

    def __getitem__(self, idx):
        r = self.rows[idx]
        messages = r['prompt_messages']
        target = r['target']

        # load images from paths
        images = [self._load_image(p) for p in r['image_paths'][:5]]

        # full conversation including assistant target
        assistant_msg = {'role': 'assistant', 'content': target}
        full_messages = messages + [assistant_msg]

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        # process full with images
        full_inputs = self.processor(text=full_text, images=images, return_tensors='pt', padding=False, truncation=False)

        # Label ONLY the target answer tokens (not the assistant turn boilerplate).
        # The target is at the very end before <end_of_turn>; find its last occurrence.
        target_ids = self.processor.tokenizer.encode(target, add_special_tokens=False)
        full_seq = full_inputs['input_ids'][0].tolist()
        labels = torch.full_like(full_inputs['input_ids'], -100)
        found = False
        for i in range(len(full_seq) - len(target_ids), -1, -1):
            if full_seq[i:i + len(target_ids)] == target_ids:
                labels[:, i:i + len(target_ids)] = full_inputs['input_ids'][:, i:i + len(target_ids)]
                found = True
                break
        if not found:
            # fallback: leave the last token unmasked so loss stays defined
            labels[:, -1] = full_inputs['input_ids'][:, -1]

        # optional left-side truncation to max_length (keep target at the end)
        seq_len = full_inputs['input_ids'].shape[1]
        if seq_len > self.max_length:
            start = seq_len - self.max_length
            for k in ['input_ids', 'attention_mask', 'labels']:
                full_inputs[k] = full_inputs[k][:, start:]
            if full_inputs.get('mm_token_type_ids') is not None:
                full_inputs['mm_token_type_ids'] = full_inputs['mm_token_type_ids'][:, start:]
            # image features are kept as-is; extra image tokens are simply ignored

        item = {
            'input_ids': full_inputs['input_ids'].squeeze(0),
            'attention_mask': full_inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'pixel_values': full_inputs.get('pixel_values', None),
            'mm_token_type_ids': full_inputs.get('mm_token_type_ids', None).squeeze(0) if full_inputs.get('mm_token_type_ids', None) is not None else None,
            'image_position_ids': full_inputs.get('image_position_ids', None),
        }
        return item


def collate_fn(batch):
    input_ids = [b['input_ids'] for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]
    labels = [b['labels'] for b in batch]

    # manual pad
    max_len = max(len(x) for x in input_ids)
    pad_id = 0
    padded_ids = []
    padded_mask = []
    padded_labels = []
    for ids, mask, lbl in zip(input_ids, attention_mask, labels):
        pad_len = max_len - len(ids)
        padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)]))
        padded_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
        padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)]))

    out = {
        'input_ids': torch.stack(padded_ids),
        'attention_mask': torch.stack(padded_mask),
        'labels': torch.stack(padded_labels),
    }

    # pixel_values: concatenate across batch (vision tower treats each image as batch item)
    pvs = [b['pixel_values'] for b in batch if b['pixel_values'] is not None]
    if pvs:
        out['pixel_values'] = torch.cat(pvs, dim=0)

        # mm_token_type_ids: pad and stack normally (per-example sequence signal)
        mm_types = [b['mm_token_type_ids'] for b in batch if b['mm_token_type_ids'] is not None]
        if mm_types:
            max_len_mm = max(t.shape[0] for t in mm_types)
            padded_mm = []
            for t in mm_types:
                if t.shape[0] < max_len_mm:
                    pad = torch.zeros(max_len_mm - t.shape[0], dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad])
                padded_mm.append(t)
            out['mm_token_type_ids'] = torch.stack(padded_mm)

        # image_position_ids: concatenate across batch along image dimension
        img_pos = [b['image_position_ids'] for b in batch if b['image_position_ids'] is not None]
        if img_pos:
            out['image_position_ids'] = torch.cat(img_pos, dim=0)
    return out


_NAN_SKIPS = {'n': 0}


class NaNProofAdamW(torch.optim.AdamW):
    """Zero non-finite grads before stepping: a rare Gemma4 backward NaN
    on specific image pairs would otherwise poison the weights."""

    def step(self, closure=None):
        n_bad = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    n_bad += 1
        if n_bad:
            _NAN_SKIPS['n'] += 1
            print(f'[nan-guard] zeroed {n_bad} grad tensors (total skipped steps: {_NAN_SKIPS["n"]})', flush=True)
        return super().step(closure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_id', type=str, required=True)
    ap.add_argument('--dataset', type=str, required=True, choices=['std', 'noocr', 'ocrlong'])
    ap.add_argument('--r', type=int, default=64)
    ap.add_argument('--alpha', type=int, default=128)
    ap.add_argument('--lora_lm_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--lora_vis_modules', type=str, default='')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--max_length', type=int, default=3072)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--warmup_ratio', type=float, default=0.05)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--no_grad_ckpt', action='store_true')
    ap.add_argument('--train_path', type=str, default=None)
    ap.add_argument('--val_path', type=str, default=None)
    args = ap.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['VLLM_PLUGINS'] = ''
    os.environ['VLLM_USE_DEEP_GEMM'] = '0'

    exp_dir = os.path.join(WORK_DIR, 'exps', args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    config = vars(args)
    config['work_dir'] = WORK_DIR
    config['gemma_dir'] = GEMMA_DIR
    with open(os.path.join(exp_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    train_path = args.train_path or os.path.join(WORK_DIR, f'sft_{args.dataset}_train.jsonl')
    val_path = args.val_path or os.path.join(WORK_DIR, f'sft_{args.dataset}_val.jsonl')

    processor = AutoProcessor.from_pretrained(GEMMA_DIR, trust_remote_code=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        GEMMA_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        trust_remote_code=True,
    )

    # print module names
    module_names = []
    for name, _ in model.named_modules():
        module_names.append(name)
    with open(os.path.join(exp_dir, 'module_names.txt'), 'w', encoding='utf-8') as f:
        for name in module_names:
            f.write(name + '\n')
    print(f'[train] total modules: {len(module_names)}')

    target_patterns = []
    if args.lora_lm_modules:
        target_patterns.extend(args.lora_lm_modules.split(','))
    if args.lora_vis_modules:
        target_patterns.extend(args.lora_vis_modules.split(','))

    target_modules = []
    for name in module_names:
        # skip vision/audio towers unless explicitly requested; Gemma4ClippableLinear wrappers not supported by peft 0.20
        if ('vision_tower' in name or 'audio_tower' in name) and not args.lora_vis_modules:
            continue
        # only language model by default
        if not args.lora_vis_modules and 'language_model' not in name:
            continue
        for pat in target_patterns:
            if name.endswith('.' + pat):
                target_modules.append(name)
                break

    print(f'[train] LoRA targets: {len(target_modules)} modules')
    with open(os.path.join(exp_dir, 'lora_targets.txt'), 'w', encoding='utf-8') as f:
        for t in target_modules:
            f.write(t + '\n')

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = QCChatDataset(train_path, processor, max_length=args.max_length)
    val_ds = QCChatDataset(val_path, processor, max_length=args.max_length)

    steps_per_epoch = len(train_ds) // (args.batch_size * args.grad_accum)
    warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, 'checkpoints'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=False,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False,
        seed=42,
        report_to='none',
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    optimizer = NaNProofAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
    )

    trainer.train()

    adapter_dir = os.path.join(exp_dir, 'adapter')
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f'[train] adapter saved to {adapter_dir}')

    hist = trainer.state.log_history
    with open(os.path.join(exp_dir, 'train_log.jsonl'), 'w', encoding='utf-8') as f:
        for entry in hist:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
