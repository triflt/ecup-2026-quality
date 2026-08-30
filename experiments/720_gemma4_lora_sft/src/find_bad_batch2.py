"""Find which micro-batch produces NaN grads (backward with gradient checkpointing)."""
import os
import sys
import argparse

import torch
from transformers import Gemma4ForConditionalGeneration, AutoProcessor

sys.path.insert(0, '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC')
from train_lora import QCChatDataset, collate_fn, GEMMA_DIR, WORK_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='noocr')
ap.add_argument('--batch_size', type=int, default=4)
ap.add_argument('--n_micros', type=int, default=170)
ap.add_argument('--start_micro', type=int, default=0)
ap.add_argument('--ids', type=str, default=None, help='comma-separated item indices to test individually')
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--max_length', type=int, default=3072)
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

processor = AutoProcessor.from_pretrained(GEMMA_DIR, trust_remote_code=True)
model = Gemma4ForConditionalGeneration.from_pretrained(
    GEMMA_DIR, torch_dtype=torch.bfloat16, attn_implementation='sdpa', trust_remote_code=True,
).cuda()

# match train_lora.py: LM-only LoRA targets
from peft import LoraConfig, get_peft_model
module_names = [name for name, _ in model.named_modules()]
target_patterns = 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'.split(',')
target_modules = []
for name in module_names:
    if 'vision_tower' in name or 'audio_tower' in name:
        continue
    if 'language_model' not in name:
        continue
    for pat in target_patterns:
        if name.endswith('.' + pat):
            target_modules.append(name)
            break
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules,
                         lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.enable_input_require_grads()
model.train()

train_path = os.path.join(WORK_DIR, f'sft_{args.dataset}_train.jsonl')
ds = QCChatDataset(train_path, processor, max_length=args.max_length)

g = torch.Generator().manual_seed(42)
sampler = list(torch.utils.data.RandomSampler(ds, generator=g))
print(f'total items {len(ds)}, scanning {args.n_micros} micro-batches of {args.batch_size}', flush=True)

if args.ids:
    target_idxs = [int(x) for x in args.ids.split(',')]
    for idx in target_idxs:
        batch = collate_fn([ds[idx]])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        print(f'idx={idx} loss={float(loss):.4f}', flush=True)
        try:
            loss.backward()
        except Exception as e:
            print(f'  idx={idx} backward EXC: {e}', flush=True)
            continue
        n_bad = sum(1 for _, p in model.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all())
        print(f'  idx={idx} nan_grad_tensors={n_bad}', flush=True)
        del out, loss, batch
        torch.cuda.empty_cache()
    print('done', flush=True)
    sys.exit(0)

for micro in range(args.start_micro, args.start_micro + args.n_micros):
    idxs = sampler[micro * args.batch_size:(micro + 1) * args.batch_size]
    batch = collate_fn([ds[i] for i in idxs])
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    opt_step = micro // 8 + 1
    print(f'opt_step={opt_step} micro={micro+1} loss={float(loss):.4f} ids={idxs}', flush=True)
    try:
        loss.backward()
    except Exception as e:
        print(f'  *** backward EXC: {e}', flush=True)
        break
    bad = []
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
        else:
            total_norm += float(p.grad.norm()) ** 2
    if bad:
        print(f'  *** NaN GRAD in {len(bad)} tensors, first: {bad[0]}', flush=True)
        print(f'  *** ids={idxs}', flush=True)
        break
    if not torch.isfinite(torch.tensor(total_norm)):
        print('  *** total norm nan', flush=True)
        break
    del out, loss, batch
    torch.cuda.empty_cache()
print('done', flush=True)
