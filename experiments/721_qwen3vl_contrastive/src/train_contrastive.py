"""Contrastive (SupCon/InfoNCE) fine-tuning of Qwen3-VL-Embedding-2B for QC."""
import os
import sys
import json
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
EMBEDDER_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-Embedding-2B'
SUBMIT_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/submit'


def label_for_class(category: str, raw_label: int) -> int:
    cat_idx = 0 if category == 'БАД' else 1
    lbl = 1 if raw_label == 1 else 0
    return cat_idx * 2 + lbl


def load_items(split_name: str):
    import pandas as pd
    df = pd.read_csv(os.path.join(WORK_DIR, 'data', 'data.csv'))
    ocr = pd.read_parquet(os.path.join(WORK_DIR, 'ocr_cache.parquet'))
    ocr_map = dict(zip(ocr['id'].astype(int), ocr['ocr_text_full']))
    with open(os.path.join(WORK_DIR, 'split.json'), 'r', encoding='utf-8') as f:
        split = json.load(f)
    df = df[df['id'].isin(split[split_name])]
    rows = []
    for _, row in df.iterrows():
        item_id = int(row['id'])
        rows.append({
            'id': item_id,
            'category': row['category'],
            'label': int(row['label']),
            'name': row['name'],
            'description': row['description'],
            'ocr_text': ocr_map.get(item_id, ''),
        })
    return rows


def build_text(item):
    sys.path.insert(0, SUBMIT_DIR)
    from src.gemma_stage import DESC_MAX, OCR_MAX
    desc = str(item.get('description') or '')[:DESC_MAX]
    ocr = str(item.get('ocr_text') or '')[:OCR_MAX]
    text = (f"Категория: {item['category']}\n"
            f"Название: {item['name']}\n"
            f"Описание: {desc}\n"
            f"Текст с изображениями товара (OCR): {ocr}")
    return text


class ContrastiveDataset(Dataset):
    def __init__(self, items, processor, images_dir, n_images=5):
        self.items = items
        self.processor = processor
        self.images_dir = Path(images_dir)
        self.n_images = n_images

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        sys.path.insert(0, SUBMIT_DIR)
        from src.ocr_stage import image_paths_for
        from PIL import Image
        images = [Image.open(p).convert('RGB') for p in image_paths_for(item['id'], self.images_dir)[:self.n_images]]
        text = build_text(item)
        cls = label_for_class(item['category'], item['label'])
        messages = [
            {'role': 'system', 'content': 'Represent the user\'s input.'},
            {'role': 'user', 'content': [{'type': 'image', 'image': im} for im in images] + [{'type': 'text', 'text': text}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_vision_id=False)
        return {
            'prompt': prompt,
            'images': images,
            'cls': torch.tensor(cls, dtype=torch.long),
        }


def collate_fn(batch):
    prompts = [b['prompt'] for b in batch]
    images = [b['images'] for b in batch]
    cls = torch.stack([b['cls'] for b in batch])
    # We need processor here; use global or pass via closure. We'll use global_train_processor.
    global _train_processor
    inputs = _train_processor(text=prompts, images=images, return_tensors='pt', padding=True, truncation=False)
    out = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'cls': cls,
    }
    for key in ('pixel_values', 'image_grid_thw', 'mm_token_type_ids', 'pixel_attention_mask', 'image_rotary_emb'):
        if key in inputs:
            out[key] = inputs[key]
    return out


class BalancedClassSampler(Sampler):
    def __init__(self, items, num_classes, samples_per_class, seed=42):
        self.items = items
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.class_to_indices = [[] for _ in range(num_classes)]
        for i, item in enumerate(items):
            cls = label_for_class(item['category'], item['label'])
            self.class_to_indices[cls].append(i)
        self.epoch = 0

    def __len__(self):
        min_count = min(len(idxs) for idxs in self.class_to_indices)
        return (min_count // self.samples_per_class) * self.num_classes * self.samples_per_class

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.epoch)
        per_class = [idxs[:] for idxs in self.class_to_indices]
        for idxs in per_class:
            rng.shuffle(idxs)
        min_count = min(len(idxs) for idxs in per_class)
        n_batches = min_count // self.samples_per_class
        indices = []
        for _ in range(n_batches):
            for c in range(self.num_classes):
                indices.extend(per_class[c][:self.samples_per_class])
                per_class[c] = per_class[c][self.samples_per_class:]
        self.epoch += 1
        return iter(indices)


def pool_last_token(hidden, attention_mask):
    last_idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    emb = hidden[batch_idx, last_idx, :]
    return F.normalize(emb, p=2, dim=1)


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, sampler=None, loss_type='sigmoid', tau=0.07,
                 hard_mining=False, margin=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_sampler = sampler
        self.loss_type = loss_type
        self.tau = tau
        self.hard_mining = hard_mining
        self.margin = margin

    def _get_train_sampler(self, train_dataset=None):
        if self.custom_sampler is not None:
            return self.custom_sampler
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cls = inputs['cls']
        model_inputs = {k: v for k, v in inputs.items() if k != 'cls'}
        outputs = model(**model_inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        emb = pool_last_token(hidden, model_inputs['attention_mask'])
        if self.loss_type == 'infonce':
            loss = supervised_infonce(emb, cls, self.tau, self.hard_mining)
        elif self.loss_type == 'triplet':
            loss = triplet_loss(emb, cls, self.margin)
        elif self.loss_type == 'sigmoid':
            loss = sigmoid_loss(emb, cls, self.tau)
        else:
            raise ValueError(f'Unknown loss_type={self.loss_type}')
        return (loss, outputs) if return_outputs else loss


def supervised_infonce(embeddings, labels, tau=0.07, hard_mining=False):
    sim = torch.matmul(embeddings, embeddings.T) / tau
    mask_self = torch.eye(sim.size(0), device=sim.device).bool()
    sim = sim.masked_fill(mask_self, -1e9)
    labels = labels.unsqueeze(0)
    positive_mask = (labels == labels.T).float()
    positive_mask = positive_mask.masked_fill(mask_self, 0.0)

    if hard_mining:
        pos_sim = sim.clone()
        pos_sim[positive_mask == 0] = -1e9
        hardest_positive, _ = pos_sim.max(dim=1)
        neg_sim = sim.clone()
        neg_sim[positive_mask == 1] = -1e9
        hardest_negative, _ = neg_sim.max(dim=1)
        loss = F.softplus(hardest_negative - hardest_positive).mean()
        return loss

    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * positive_mask).sum(dim=1)
    neg_sum = (exp_sim * (1 - positive_mask)).sum(dim=1)
    loss = -torch.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))
    loss = loss[positive_mask.sum(dim=1) > 0].mean()
    return loss


def triplet_loss(embeddings, labels, margin=0.3):
    """Online triplet mining with cosine similarity (normalized embeddings)."""
    sim = torch.matmul(embeddings, embeddings.T)  # cosine similarity
    mask_self = torch.eye(sim.size(0), device=sim.device).bool()
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)

    # hardest positive: max similarity among same class (exclude self)
    pos_sim = sim.masked_fill(~labels_eq | mask_self, -1e9)
    hardest_positive, _ = pos_sim.max(dim=1)

    # hardest negative: min similarity among different class
    neg_sim = sim.masked_fill(labels_eq | mask_self, 1e9)
    hardest_negative, _ = neg_sim.min(dim=1)

    loss = F.relu(hardest_negative - hardest_positive + margin)
    valid = (pos_sim != -1e9).any(dim=1) & (neg_sim != 1e9).any(dim=1)
    loss = loss[valid].mean()
    return loss


def sigmoid_loss(embeddings, labels, tau=0.07):
    """Sigmoid-style multi-positive contrastive loss.
    For each anchor, positives are pushed together and negatives apart
    via a softmax over pairwise similarities. Works with small/mixed batches.
    """
    sim = torch.matmul(embeddings, embeddings.T) / tau
    mask_self = torch.eye(sim.size(0), device=sim.device).bool()
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = labels_eq & ~mask_self

    # target distribution: uniform over positives for each anchor
    target = positive_mask.float()
    pos_counts = target.sum(dim=1, keepdim=True)
    target = target / (pos_counts + 1e-8)

    log_probs = F.log_softmax(sim, dim=1)
    loss = -(target * log_probs).sum(dim=1)
    loss = loss[pos_counts.squeeze(1) > 0].mean()
    return loss


def main():
    global _train_processor
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_id', type=str, required=True)
    ap.add_argument('--r', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--samples_per_class', type=int, default=2)
    ap.add_argument('--loss_type', type=str, default='sigmoid', choices=['sigmoid', 'infonce', 'triplet'])
    ap.add_argument('--tau', type=float, default=0.07)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--hard_mining', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_length', type=int, default=4096)
    ap.add_argument('--n_images', type=int, default=2)
    args = ap.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    exp_dir = os.path.join(WORK_DIR, 'exps', args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    config = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    processor = AutoProcessor.from_pretrained(EMBEDDER_DIR, trust_remote_code=True)
    _train_processor = processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(EMBEDDER_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True)

    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    lora_config = LoraConfig(r=args.r, lora_alpha=args.alpha, target_modules=target_modules,
                             lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_items = load_items('train')
    val_items = load_items('val')

    train_ds = ContrastiveDataset(train_items, processor, os.path.join(WORK_DIR, 'data', 'images', 'images'), n_images=args.n_images)
    val_ds = ContrastiveDataset(val_items, processor, os.path.join(WORK_DIR, 'data', 'images', 'images'), n_images=args.n_images)

    sampler = BalancedClassSampler(train_items, num_classes=4, samples_per_class=args.samples_per_class, seed=42)

    training_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, 'checkpoints_embed'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False,
        seed=42,
        report_to='none',
        dataloader_num_workers=0,
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        sampler=sampler,
        loss_type=args.loss_type,
        tau=args.tau,
        hard_mining=args.hard_mining,
        margin=args.margin,
    )
    trainer.train()

    adapter_dir = os.path.join(exp_dir, 'adapter_embed')
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f'[train] embed adapter saved to {adapter_dir}')


if __name__ == '__main__':
    main()
