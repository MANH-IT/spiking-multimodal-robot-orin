import torch, sys
sys.path.insert(0, '.')
ckpt = torch.load('vision_system/weights/finetuned/checkpoint_latest.pth',
                   map_location='cpu', weights_only=False)
print("=== Checkpoint Info ===")
print("Epoch:", ckpt["epoch"])
print("Val loss:", round(ckpt["val_loss"], 6))

# Count classes from cls_head output dim
for k, v in ckpt["model_state_dict"].items():
    if "cls_head" in k or "bbox_head" in k:
        print(f"  {k}: {list(v.shape)}")

# Now check what num_classes the dataset would produce
from vision_system.data.rgbd_sequence_with_gt_dataset import HILORawDataset, DEFAULT_HILO_ROOT
ds = HILORawDataset(
    hilo_root=DEFAULT_HILO_ROOT,
    img_size=224,
    sequence_length=4,
    arc_ids=[0,1,2,3],
    max_scenes=None,
)
print(f"\nDataset num_classes: {ds.num_classes}")
print(f"Dataset samples: {len(ds)}")
