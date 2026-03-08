# -*- coding: utf-8 -*-
"""
NLP Training Script - NCKH 2026
================================
Training pipeline hoàn chỉnh cho SpikingLanguageModel với:
- 80/10/10 Train/Val/Test split
- Validation loop + best model checkpoint
- Early stopping (patience=5)
- LR Scheduler (ReduceLROnPlateau)
- Spike rate monitoring (energy metric)
- Per-language accuracy tracking
- Detailed logging

Usage:
    python nlp_system/train_nlp.py
    python nlp_system/train_nlp.py --epochs 20 --batch-size 64 --lr 5e-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# --- Add project root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nlp_system.models.spiking_language_model import SpikingLanguageModel
from nlp_system.data.vmmd_loader import VMMDDataset, collate_fn

# ─── Optional: tqdm progress bar ───────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:
        def __init__(self, iterable, **kw):
            self.it = iterable
        def __iter__(self):
            return iter(self.it)
        def set_postfix(self, **kw):
            pass
        def set_description(self, s):
            pass


# ============================================================
# Configuration
# ============================================================

def get_default_config() -> dict:
    return {
        # Data
        'data_path': str(PROJECT_ROOT / 'nlp_system' / 'data' / 'raw' / 'vmmd_large_synthetic.jsonl'),
        'train_ratio': 0.80,
        'val_ratio':   0.10,
        'test_ratio':  0.10,
        # Model
        'vocab_size': 50_000,
        'embed_dim':  256,
        'hidden_dim': 512,
        'num_steps':  25,     # SNN time steps
        'dropout':    0.3,
        # Training
        'batch_size': 64,
        'epochs':     15,
        'lr':         5e-4,
        'weight_decay':     1e-4,
        'patience':         5,     # Early stopping
        'lr_patience':      2,     # LR scheduler patience
        'lr_factor':        0.5,
        'grad_clip':        1.0,
        'context_loss_weight': 0.2,  # Weight của context classification loss
        # Paths
        'output_dir': str(PROJECT_ROOT / 'experiments' / 'nlp_training'),
        'model_save':  str(PROJECT_ROOT / 'nlp_system' / 'models' / 'nlp_model.pth'),
        'intent_map_path': str(PROJECT_ROOT / 'nlp_system' / 'models' / 'intent_mapping.json'),
        'context_map_path': str(PROJECT_ROOT / 'nlp_system' / 'models' / 'context_mapping.json'),
    }


# ============================================================
# Helpers
# ============================================================

def build_label_maps(data_path: str) -> tuple[dict, dict, dict, dict]:
    """
    Quét file 1 lần → xây dựng intent và context maps.
    Returns: intent_to_idx, idx_to_intent, context_to_idx, idx_to_context
    """
    intents, contexts = set(), set()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                intents.add(item['intent'])
                contexts.add(item.get('context', 'home'))
            except (json.JSONDecodeError, KeyError):
                continue

    intent_list  = sorted(intents)
    context_list = sorted(contexts)

    intent_to_idx  = {v: i for i, v in enumerate(intent_list)}
    idx_to_intent  = {i: v for v, i in intent_to_idx.items()}
    context_to_idx = {v: i for i, v in enumerate(context_list)}
    idx_to_context = {i: v for v, i in context_to_idx.items()}

    return intent_to_idx, idx_to_intent, context_to_idx, idx_to_context


def validate(
    model: SpikingLanguageModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    intent_to_idx: dict,
    context_to_idx: dict,
    context_loss_weight: float = 0.2,
) -> dict:
    """Validation loop — returns metrics dict."""
    model.eval()
    total_loss = correct_intent = correct_context = total = 0
    lang_correct: dict[str, int] = {}
    lang_total:   dict[str, int] = {}
    spike_rates = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            langs  = batch['language']

            targets_intent = torch.tensor(
                [intent_to_idx[i] for i in batch['intent']],
                dtype=torch.long, device=device,
            )
            targets_context = torch.tensor(
                [context_to_idx.get(c, 0) for c in batch['context']],
                dtype=torch.long, device=device,
            )

            out = model(inputs, mask)
            loss_intent  = criterion(out['intent_logits'], targets_intent)
            loss_context = criterion(out['context_logits'], targets_context)
            loss = loss_intent + context_loss_weight * loss_context

            total_loss += loss.item() * len(inputs)
            total      += len(inputs)
            spike_rates.append(out['spike_rate'])

            preds_intent  = out['intent_logits'].argmax(dim=-1)
            preds_context = out['context_logits'].argmax(dim=-1)
            correct_intent  += (preds_intent  == targets_intent).sum().item()
            correct_context += (preds_context == targets_context).sum().item()

            # Per-language accuracy
            for i, lang in enumerate(langs):
                if lang not in lang_total:
                    lang_correct[lang] = 0
                    lang_total[lang]   = 0
                lang_total[lang] += 1
                if preds_intent[i] == targets_intent[i]:
                    lang_correct[lang] += 1

    avg_loss   = total_loss / max(total, 1)
    intent_acc = correct_intent / max(total, 1) * 100
    ctx_acc    = correct_context / max(total, 1) * 100
    avg_spike  = sum(spike_rates) / max(len(spike_rates), 1)
    lang_accs  = {
        lang: lang_correct[lang] / max(lang_total[lang], 1) * 100
        for lang in lang_total
    }

    return {
        'loss':        avg_loss,
        'intent_acc':  intent_acc,
        'context_acc': ctx_acc,
        'spike_rate':  avg_spike,
        'lang_accs':   lang_accs,
    }


def log(msg: str, log_file=None):
    """Print + write to log file."""
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


# ============================================================
# Main Training Function
# ============================================================

def train(cfg: dict):
    # ── Setup ──────────────────────────────────────────────
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'train.log'
    ckpt_best = out_dir / 'best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n{'='*60}", log_file)
    log(f"  NLP Training — SpikingLanguageModel (NCKH 2026)", log_file)
    log(f"{'='*60}", log_file)
    log(f"  Device  : {device}", log_file)
    log(f"  Data    : {cfg['data_path']}", log_file)

    # ── Build label maps ────────────────────────────────────
    log("\n[1/4] Building label maps...", log_file)
    intent_to_idx, idx_to_intent, context_to_idx, idx_to_context = \
        build_label_maps(cfg['data_path'])

    output_dim = len(intent_to_idx)
    log(f"  Intents  : {output_dim} → {list(intent_to_idx.keys())}", log_file)
    log(f"  Contexts : {len(context_to_idx)} → {list(context_to_idx.keys())}", log_file)

    # Save maps
    with open(cfg['intent_map_path'], 'w', encoding='utf-8') as f:
        json.dump(intent_to_idx, f, ensure_ascii=False, indent=2)
    with open(cfg['context_map_path'], 'w', encoding='utf-8') as f:
        json.dump(context_to_idx, f, ensure_ascii=False, indent=2)

    # ── Dataset + Split ─────────────────────────────────────
    log("\n[2/4] Loading dataset...", log_file)
    full_dataset = VMMDDataset(cfg['data_path'])
    total_size = len(full_dataset)
    train_size = int(total_size * cfg['train_ratio'])
    val_size   = int(total_size * cfg['val_ratio'])
    test_size  = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    log(f"  Total   : {total_size:,}", log_file)
    log(f"  Train   : {train_size:,} ({cfg['train_ratio']*100:.0f}%)", log_file)
    log(f"  Val     : {val_size:,} ({cfg['val_ratio']*100:.0f}%)", log_file)
    log(f"  Test    : {test_size:,} ({cfg['test_ratio']*100:.0f}%)", log_file)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                               shuffle=True, collate_fn=collate_fn,
                               num_workers=0, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'] * 2,
                               shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'] * 2,
                               shuffle=False, collate_fn=collate_fn)

    # ── Model ────────────────────────────────────────────────
    log("\n[3/4] Initializing model...", log_file)
    model = SpikingLanguageModel(
        vocab_size=cfg['vocab_size'],
        embed_dim=cfg['embed_dim'],
        hidden_dim=cfg['hidden_dim'],
        output_dim=output_dim,
        num_steps=cfg['num_steps'],
        dropout=cfg['dropout'],
    ).to(device)
    log(f"  {model}", log_file)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',       # Maximize val accuracy
        factor=cfg['lr_factor'],
        patience=cfg['lr_patience'],
        verbose=True,
    )

    # ── Training Loop ────────────────────────────────────────
    log(f"\n[4/4] Training for {cfg['epochs']} epochs...", log_file)
    log(f"  Batch size    : {cfg['batch_size']}", log_file)
    log(f"  Learning rate : {cfg['lr']}", log_file)
    log(f"  Early stopping: patience={cfg['patience']}", log_file)
    log("", log_file)

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start = time.time()
        model.train()

        train_loss = train_correct = train_total = 0
        train_spike_rates = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:>2}/{cfg['epochs']}",
                    ncols=90)

        for batch in pbar:
            inputs = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)

            targets_intent = torch.tensor(
                [intent_to_idx[i] for i in batch['intent']],
                dtype=torch.long, device=device,
            )
            targets_context = torch.tensor(
                [context_to_idx.get(c, 0) for c in batch['context']],
                dtype=torch.long, device=device,
            )

            optimizer.zero_grad()
            out = model(inputs, mask)

            loss_intent  = criterion(out['intent_logits'], targets_intent)
            loss_context = criterion(out['context_logits'], targets_context)
            loss = loss_intent + cfg['context_loss_weight'] * loss_context

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()

            train_loss    += loss.item() * len(inputs)
            train_total   += len(inputs)
            preds          = out['intent_logits'].argmax(dim=-1)
            train_correct += (preds == targets_intent).sum().item()
            train_spike_rates.append(out['spike_rate'])

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{train_correct/train_total*100:.1f}%",
                spike=f"{out['spike_rate']:.3f}",
            )

        # Validation
        val_metrics = validate(
            model, val_loader, criterion, device,
            intent_to_idx, context_to_idx, cfg['context_loss_weight'],
        )

        # Stats
        avg_train_loss = train_loss / max(train_total, 1)
        avg_train_acc  = train_correct / max(train_total, 1) * 100
        avg_train_spike = sum(train_spike_rates) / max(len(train_spike_rates), 1)
        epoch_time = time.time() - epoch_start
        cur_lr = optimizer.param_groups[0]['lr']

        row = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc':  avg_train_acc,
            'val_loss':   val_metrics['loss'],
            'val_acc':    val_metrics['intent_acc'],
            'val_ctx_acc': val_metrics['context_acc'],
            'spike_rate': avg_train_spike,
            'lr': cur_lr,
        }
        history.append(row)

        log(
            f"Epoch {epoch:>2}/{cfg['epochs']} | "
            f"Train Loss={avg_train_loss:.4f} Acc={avg_train_acc:.1f}% | "
            f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['intent_acc']:.1f}% "
            f"Ctx={val_metrics['context_acc']:.1f}% | "
            f"Spike={avg_train_spike:.3f} | "
            f"LR={cur_lr:.2e} | "
            f"Time={epoch_time:.1f}s",
            log_file,
        )

        # Per-language accuracy
        for lang, acc in val_metrics.get('lang_accs', {}).items():
            log(f"         [{lang}] Val Acc={acc:.1f}%", log_file)

        # LR Scheduler
        scheduler.step(val_metrics['intent_acc'])

        # Best model checkpoint
        if val_metrics['intent_acc'] > best_val_acc:
            best_val_acc = val_metrics['intent_acc']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'intent_to_idx': intent_to_idx,
                'context_to_idx': context_to_idx,
                'config': cfg,
            }, ckpt_best)
            log(f"  ✅ Best model saved (val_acc={best_val_acc:.2f}%)", log_file)
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                log(f"\n⏹ Early stopping after {epoch} epochs "
                    f"(no improvement for {cfg['patience']} epochs).", log_file)
                break

    # ── Final Test Evaluation ───────────────────────────────
    log(f"\n{'='*60}", log_file)
    log("  Final Test Evaluation", log_file)
    log(f"{'='*60}", log_file)

    # Load best model
    ckpt = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = validate(
        model, test_loader, criterion, device,
        intent_to_idx, context_to_idx, cfg['context_loss_weight'],
    )

    log(f"  Test Intent Acc  : {test_metrics['intent_acc']:.2f}%", log_file)
    log(f"  Test Context Acc : {test_metrics['context_acc']:.2f}%", log_file)
    log(f"  Test Loss        : {test_metrics['loss']:.4f}", log_file)
    log(f"  Avg Spike Rate   : {test_metrics['spike_rate']:.4f}", log_file)

    # Chỉ tiêu đề cương: > 90% tiếng Việt, > 92% tiếng Anh, > 88% tiếng Trung
    targets = {'vi': 90.0, 'en': 92.0, 'zh': 88.0}
    log(f"\n  Per-language accuracy vs targets:", log_file)
    for lang, lang_acc in test_metrics.get('lang_accs', {}).items():
        target = targets.get(lang, 85.0)
        status = "✅" if lang_acc >= target else "❌"
        log(f"    [{lang}] {lang_acc:.1f}% {status} (target: {target}%)", log_file)

    # Save final model (dùng cho inference)
    torch.save(model.state_dict(), cfg['model_save'])
    log(f"\n  Model saved → {cfg['model_save']}", log_file)

    # Save training history
    hist_path = out_dir / 'history.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    log(f"  History saved → {hist_path}", log_file)
    log(f"\n  Best Val Acc: {best_val_acc:.2f}%", log_file)
    log(f"{'='*60}\n", log_file)

    return model, history


# ============================================================
# Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SpikingLanguageModel - NCKH 2026'
    )
    cfg = get_default_config()
    parser.add_argument('--data-path', default=cfg['data_path'])
    parser.add_argument('--epochs', type=int, default=cfg['epochs'])
    parser.add_argument('--batch-size', type=int, default=cfg['batch_size'])
    parser.add_argument('--lr', type=float, default=cfg['lr'])
    parser.add_argument('--hidden-dim', type=int, default=cfg['hidden_dim'])
    parser.add_argument('--num-steps', type=int, default=cfg['num_steps'])
    parser.add_argument('--patience', type=int, default=cfg['patience'])
    parser.add_argument('--output-dir', default=cfg['output_dir'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_default_config()

    # Override defaults với CLI args
    cfg.update({
        'data_path':  args.data_path,
        'epochs':     args.epochs,
        'batch_size': args.batch_size,
        'lr':         args.lr,
        'hidden_dim': args.hidden_dim,
        'num_steps':  args.num_steps,
        'patience':   args.patience,
        'output_dir': args.output_dir,
    })

    train(cfg)
