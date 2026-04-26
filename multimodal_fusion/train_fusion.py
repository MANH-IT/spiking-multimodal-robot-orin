"""
Fusion Training Script
=======================
Fine-tune SpikingFusionTransformer trên collected data.

Data format hỗ trợ:
  1. Synthetic (tự sinh từ training_data.json + mock SNN detections)
  2. Real    (logs từ robot thật: frame_buffer + text + action_label)

Usage:
    python multimodal_fusion/train_fusion.py
    python multimodal_fusion/train_fusion.py --epochs 30 --use-phobert
    python multimodal_fusion/train_fusion.py --data data/robot_logs.json --real
"""

import os
import sys
import json
import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Project path ─────────────────────────────────────────────────────────────
root = str(Path(__file__).parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

from multimodal_fusion.spiking_fusion import (
    SpikingFusionTransformer, build_fusion_model,
    ROBOT_ACTIONS, NLU_INTENTS, NUM_ACTIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FusionTrainer")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intent → Action mapping (ground truth cho synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

# NLU intent (từ tiếng Việt) → robot action idx
INTENT_TO_ACTION = {
    # Khi có người trong scene
    "chao":      3,  # approach_and_greet
    "theo":      1,  # follow
    "tranh":     2,  # navigate_around
    "tim":       4,  # search

    # Câu hỏi thông tin (UTC domain)
    "thong_tin_truong": 5,  # respond (RAG)
    "tuyen_sinh":        5,
    "dao_tao":           5,
    "nghien_cuu":        5,
    "khac":              0,  # idle
}

# Rule để map text → action label cho synthetic training
def text_to_action_label(text: str, has_person: bool, has_obstacle: bool) -> int:
    t = text.lower()
    if any(w in t for w in ["theo", "follow", "đi theo", "dẫn đường"]):
        return 1 if has_person else 4  # follow | search
    if any(w in t for w in ["tránh", "avoid", "né", "vật cản"]):
        return 2 if has_obstacle else 0
    if any(w in t for w in ["chào", "hello", "hi", "xin chào"]):
        return 3 if has_person else 0
    if any(w in t for w in ["tìm", "tìm kiếm", "search"]):
        return 4
    if any(w in t for w in ["thông tin", "học phí", "ngành", "tuyển sinh",
                              "phòng", "tầng", "thư viện", "nghiên cứu", "đề tài"]):
        return 5  # respond (RAG)
    return 0  # idle


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticFusionDataset(Dataset):
    """
    Sinh dữ liệu tổng hợp từ training_data.json.
    
    Mỗi mẫu:
      - vision_spikes: (T, C, H, W) tensor ngẫu nhiên (mô phỏng SNN output)
      - text_input:    (vocab_dim,) → token ids hoặc mock embedding
      - action_label:  int ∈ [0, NUM_ACTIONS-1]
      - has_person:    bool (xác định label context)
      - has_obstacle:  bool
    """

    def __init__(
        self,
        training_json: str,
        T: int = 8,
        vision_channels: int = 64,
        vision_hw: int = 16,
        n_text_tokens: int = 16,
        augment: bool = True,
        max_samples: int = None,
    ):
        self.T = T
        self.C = vision_channels
        self.HW = vision_hw
        self.n_text_tokens = n_text_tokens
        self.augment = augment

        # Load text data
        with open(training_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Bổ sung thêm robot command examples (follow/avoid/greet)
        robot_commands = [
            {"text": "đi theo người phía trước",  "intent": "theo"},
            {"text": "theo dõi người đó đi",      "intent": "theo"},
            {"text": "follow that person",         "intent": "theo"},
            {"text": "tránh vật cản bên trái",    "intent": "tranh"},
            {"text": "avoid the obstacle",         "intent": "tranh"},
            {"text": "né vật cản ngay",            "intent": "tranh"},
            {"text": "xin chào người dùng",        "intent": "chao"},
            {"text": "hello chào buổi sáng",       "intent": "chao"},
            {"text": "tìm người dùng",             "intent": "tim"},
            {"text": "tìm kiếm xung quanh",       "intent": "tim"},
            {"text": "search for people",          "intent": "tim"},
            {"text": "dừng lại",                   "intent": "khac"},
            {"text": "stop robot",                 "intent": "khac"},
        ] * 20  # repeat để cân bằng với UTC domain data

        self.data = raw + robot_commands
        if max_samples:
            self.data = self.data[:max_samples]

        # Simple vocab từ RuleBasedTextEncoder
        from multimodal_fusion.spiking_fusion import RuleBasedTextEncoder
        self.encoder = RuleBasedTextEncoder(d_model=256, n_tokens=n_text_tokens)

        log.info(f"Dataset: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item["text"]

        # Random scene: có người hoặc vật cản không?
        has_person   = random.random() > 0.4
        has_obstacle = random.random() > 0.6

        # Action label
        action_label = text_to_action_label(text, has_person, has_obstacle)

        # Mock SNN vision spikes (T, C, H, W)
        # Simulate: nếu có người → spike rate cao ở channel 0-31
        spikes = torch.rand(self.T, self.C, self.HW, self.HW) * 0.3
        if has_person:
            spikes[:, :32, 4:12, 4:12] += 0.7 * torch.rand_like(spikes[:, :32, 4:12, 4:12])
        if has_obstacle:
            spikes[:, 32:, 8:, :8] += 0.5 * torch.rand_like(spikes[:, 32:, 8:, :8])
        spikes = spikes.clamp(0, 1)

        if self.augment:
            # Augment: random noise + temporal jitter
            spikes = spikes + torch.randn_like(spikes) * 0.05
            spikes = spikes.clamp(0, 1)
            # Random temporal permutation (T=8)
            if random.random() > 0.7:
                perm_end = random.randint(6, 8)
                perm = torch.randperm(perm_end).tolist() + list(range(perm_end, self.T))
                spikes = spikes[perm]

        # Text → token IDs
        token_ids = self.encoder.encode_text(text)  # (n_text_tokens,)

        return {
            "vision_spikes": spikes,              # (T, C, H, W)
            "token_ids":     token_ids,           # (n_text_tokens,)
            "action_label":  torch.tensor(action_label, dtype=torch.long),
            "has_person":    torch.tensor(has_person,   dtype=torch.float),
            "has_obstacle":  torch.tensor(has_obstacle, dtype=torch.float),
            "text":          text,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trainer
# ─────────────────────────────────────────────────────────────────────────────

class FusionTrainer:
    """
    Training loop cho SpikingFusionTransformer.
    
    Sử dụng:
      - CrossEntropyLoss  cho action classification
      - BCE confidence loss (học phân biệt khi nào cần xác nhận)
    """

    def __init__(
        self,
        model: SpikingFusionTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        device: str = "auto",
        save_dir: str = "models/fusion",
        confidence_thr: float = 0.6,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_thr = confidence_thr

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        log.info(f"Trainer device: {self.device}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Model params: {total_params:,}")

    def _compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        vision = batch["vision_spikes"].to(self.device)   # (B, T, C, H, W)
        # DataLoader stacks thành (B, T, C, H, W) → cần (T, B, C, H, W)
        vision = vision.permute(1, 0, 2, 3, 4)           # (T, B, C, H, W)
        token_ids  = batch["token_ids"].to(self.device)   # (B, n_tok)
        action_lbl = batch["action_label"].to(self.device) # (B,)

        out = self.model(vision, token_ids)

        # Action classification loss
        loss_action = self.ce_loss(out["action_logits"], action_lbl)

        # Confidence loss: confidence cao nếu action == label, thấp nếu sai
        with torch.no_grad():
            correct = (out["action_idx"] == action_lbl).float().unsqueeze(-1)
        loss_conf = F.binary_cross_entropy(out["confidence"], correct)

        loss = loss_action + 0.3 * loss_conf

        metrics = {
            "loss":        loss.item(),
            "loss_action": loss_action.item(),
            "loss_conf":   loss_conf.item(),
            "acc":         (out["action_idx"] == action_lbl).float().mean().item(),
        }
        return loss, metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals = {}

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in metrics.items():
                totals[k] = totals.get(k, 0) + v

            if (i + 1) % 50 == 0:
                step_metrics = {k: v / (i + 1) for k, v in totals.items()}
                log.info(
                    f"Epoch {epoch} [{i+1}/{len(self.train_loader)}] "
                    f"loss={step_metrics['loss']:.4f} "
                    f"acc={step_metrics['acc']:.3f}"
                )

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if not self.val_loader:
            return {}
        self.model.eval()
        totals = {}
        for batch in self.val_loader:
            _, metrics = self._compute_loss(batch)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0) + v
        n = len(self.val_loader)
        return {f"val_{k}": v / n for k, v in totals.items()}

    def fit(self, epochs: int = 30) -> List[Dict]:
        history = []
        best_acc = 0.0
        best_path = self.save_dir / "best_fusion.pth"

        for epoch in range(1, epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m   = self.validate()
            self.scheduler.step()

            combined = {**train_m, **val_m, "epoch": epoch}
            history.append(combined)

            acc = val_m.get("val_acc", train_m.get("acc", 0))
            log.info(
                f"[Epoch {epoch:3d}] "
                f"loss={train_m['loss']:.4f}  "
                f"acc={train_m['acc']:.3f}  "
                + (f"val_acc={val_m['val_acc']:.3f}" if val_m else "")
            )

            if acc >= best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), best_path)
                log.info(f"   💾 Saved best model (acc={best_acc:.3f})")

        log.info(f"\n✅ Training complete. Best acc: {best_acc:.3f}")
        log.info(f"   Model saved to: {best_path}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Multimodal Fusion Transformer")
    p.add_argument("--data",        default="data/training_data.json")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--d-model",     type=int,   default=256)
    p.add_argument("--num-layers",  type=int,   default=3)
    p.add_argument("--num-heads",   type=int,   default=4)
    p.add_argument("--T",           type=int,   default=8)
    p.add_argument("--val-split",   type=float, default=0.15)
    p.add_argument("--use-phobert", action="store_true",
                   help="Dùng PhoBERT CLS embedding (cần transformers)")
    p.add_argument("--save-dir",    default="models/fusion")
    p.add_argument("--max-samples", type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Dataset
    data_path = os.path.join(root, args.data)
    dataset = SyntheticFusionDataset(
        training_json=data_path,
        T=args.T,
        augment=True,
        max_samples=args.max_samples,
    )

    # Train/Val split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = build_fusion_model({
        "vision_channels": 64,
        "text_dim":        768,
        "d_model":         args.d_model,
        "num_heads":       args.num_heads,
        "num_layers":      args.num_layers,
        "T":               args.T,
        "use_phobert":     args.use_phobert,
        "confidence_thr":  0.6,
    })

    # Train
    trainer = FusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        save_dir=os.path.join(root, args.save_dir),
    )
    history = trainer.fit(epochs=args.epochs)

    # Save history
    hist_path = os.path.join(root, args.save_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"History saved to {hist_path}")


if __name__ == "__main__":
    main()
