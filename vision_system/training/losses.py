import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_box=5.0, lambda_conf=1.0, lambda_class=1.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_conf = lambda_conf
        self.lambda_class = lambda_class
        
    def forward(self, pred_boxes, pred_conf, pred_class, target_boxes, target_conf, target_class):
        # pred_boxes: (B, N, 4), target_boxes: (B, N, 4), chỉ tính trên ô có obj
        mask = (target_conf == 1).float()
        
        # Hỗ trợ batch matrix (B, N, 4)
        box_loss = F.smooth_l1_loss(pred_boxes * mask, target_boxes * mask, reduction='sum')
        box_loss = box_loss / (mask.sum() + 1e-6)
        
        conf_loss = F.binary_cross_entropy(pred_conf, target_conf, reduction='mean')
        class_loss = F.cross_entropy(pred_class.permute(0,2,1), target_class.long(), reduction='mean')
        
        total = self.lambda_box * box_loss + self.lambda_conf * conf_loss + self.lambda_class * class_loss
        return total, box_loss, conf_loss, class_loss
