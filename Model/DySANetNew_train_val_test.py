import os
import gc 
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    mean_absolute_error
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from _5DataLoder import get_dataloaders
from DySANet import UNet   

def dice_coef_binary(preds_prob, targets, threshold=0.5, smooth=1e-6):
    preds = (preds_prob > threshold).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * inter + smooth) / (union + smooth)
    return dice.mean()

def iou_coef_binary(preds_prob, targets, threshold=0.5, smooth=1e-6):
    preds = (preds_prob > threshold).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))
    iou = (inter + smooth) / (union + smooth)
    return iou.mean()

class DiceLossBinary(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLossBinary, self).__init__()
        self.smooth = smooth

    def forward(self, inputs_logits, targets):
        inputs_prob = torch.sigmoid(inputs_logits)
        inter = (inputs_prob * targets).sum(dim=(1,2,3))
        union = inputs_prob.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice_loss = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        return dice_loss.mean()

def compute_metrics_single(y_true, y_pred):
    coords_true = np.array(np.where(y_true == 1)).T
    coords_pred = np.array(np.where(y_pred == 1)).T
    
    if len(coords_true) == 0 and len(coords_pred) == 0:
        hsd = 0.0
    elif len(coords_true) == 0 or len(coords_pred) == 0:
        hsd = np.sqrt(y_true.shape[0]**2 + y_true.shape[1]**2) if len(y_true.shape) >= 2 else 0.0
    else:
        hsd = max(directed_hausdorff(coords_true, coords_pred)[0], 
                  directed_hausdorff(coords_pred, coords_true)[0])

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    acc  = accuracy_score(y_true_f, y_pred_f)
    prec = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec  = recall_score(y_true_f, y_pred_f, zero_division=0)
    dice = f1_score(y_true_f, y_pred_f, zero_division=0)
    iou  = jaccard_score(y_true_f, y_pred_f, zero_division=0)
    mae  = mean_absolute_error(y_true_f, y_pred_f)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Dice": dice,
        "IoU": iou,
        "HSD": hsd,
        "MAE": mae
    }

def visualize_and_save(img_tensor, mask_tensor, pred_mask, edge_gt_tensor, edge_pred_tensor, save_path, idx):
    import matplotlib
    matplotlib.use('Agg')
    img = img_tensor.numpy().transpose(1,2,0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    mask = mask_tensor.numpy()
    pred = pred_mask.numpy()
    
    edge_gt = edge_gt_tensor.numpy()
    edge_pred = edge_pred_tensor.numpy()
    
    img_to_save = img if img.shape[2]==3 else img[:,:,0]
    
    plt.imsave(os.path.join(save_path, f"img_{idx}.png"), img_to_save, cmap='gray' if img.shape[2]!=3 else None)
    plt.imsave(os.path.join(save_path, f"mask_gt_{idx}.png"), mask, cmap='gray')
    plt.imsave(os.path.join(save_path, f"mask_pred_{idx}.png"), pred, cmap='gray')
    plt.imsave(os.path.join(save_path, f"edge_gt_{idx}.png"), edge_gt, cmap='gray')
    plt.imsave(os.path.join(save_path, f"edge_pred_{idx}.png"), edge_pred, cmap='gray')

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(img_to_save, cmap='gray')
    axs[0].set_title("Image"); axs[0].axis('off')
    
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("GT Mask"); axs[1].axis('off')
    
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Pred Mask"); axs[2].axis('off')
    
    axs[3].imshow(edge_gt, cmap='gray')
    axs[3].set_title("GT Edge"); axs[3].axis('off')
    
    axs[4].imshow(edge_pred, cmap='gray')
    axs[4].set_title("Pred Edge"); axs[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"vis_{idx}.png"))
    plt.close()

def kl_divergence_loss(mu, logvar):
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

def train_one_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    
    bce_loss_func = nn.BCEWithLogitsLoss()
    dice_loss_func = DiceLossBinary() 

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for imgs, masks, edge_masks in pbar:
        imgs  = imgs.to(device)
        
        masks = masks.float().to(device) 
        if masks.dim() == 3:
            masks = masks.unsqueeze(1) 
            
        edge_masks = edge_masks.float().to(device) 
        if edge_masks.dim() == 3:
            edge_masks = edge_masks.unsqueeze(1) 

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out4_logits, out3_logits, out2_logits, out1_logits = model(imgs)

            target_224 = masks

            out_1_up = F.interpolate(out1_logits, size=(224, 224), mode='bilinear', align_corners=False)
            out_2_up = F.interpolate(out2_logits, size=(224, 224), mode='bilinear', align_corners=False)
            out_3_up = F.interpolate(out3_logits, size=(224, 224), mode='bilinear', align_corners=False)
            out_4_up = F.interpolate(out4_logits, size=(224, 224), mode='bilinear', align_corners=False)

            loss_seg_1 = 0.5 * bce_loss_func(out_1_up, target_224)  + 0.5 * dice_loss_func(out_1_up, target_224)
            loss_seg_2 = 0.5 * bce_loss_func(out_2_up, target_224)  + 0.5 * dice_loss_func(out_2_up, target_224)
            loss_seg_3 = 0.5 * bce_loss_func(out_3_up, target_224)  + 0.5 * dice_loss_func(out_3_up, target_224)
            loss_seg_4 = 0.5 * bce_loss_func(out_4_up, target_224)  + 0.5 * dice_loss_func(out_4_up, target_224)

            loss_seg_total = (
                1 * loss_seg_1 + 
                0 * loss_seg_2 + 
                0 * loss_seg_3 +
                0 * loss_seg_4
            )
            
            loss = loss_seg_total

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

        preds_prob = torch.sigmoid(out1_logits)
        dice = dice_coef_binary(preds_prob, masks)
        iou  = iou_coef_binary(preds_prob, masks)
        
        running_dice += dice.item() * imgs.size(0)
        running_iou  += iou.item() * imgs.size(0)

        pbar.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_dice = running_dice / len(train_loader.dataset)
    epoch_iou  = running_iou  / len(train_loader.dataset)
    return epoch_loss, epoch_dice, epoch_iou

def validate_one_epoch(model, val_loader, device):
    model.eval()
    val_metrics_list = []
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for imgs, masks, edge_masks in pbar:
            imgs = imgs.to(device)
            
            masks = masks.float().to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)      
            
            out4_logits, out3_logits, out2_logits, out1_logits = model(imgs)
            
            preds_prob = torch.sigmoid(out1_logits)
            preds = (preds_prob > 0.5).float()
            
            for i in range(imgs.size(0)):
                m = compute_metrics_single(
                    masks[i].squeeze(0).cpu().numpy(),
                    preds[i].squeeze(0).cpu().numpy()
                )
                val_metrics_list.append(m)
    
    epoch_loss = 0.0 
    
    df_val = pd.DataFrame(val_metrics_list)
    mean_val_metrics = df_val.mean().to_dict()
    
    return epoch_loss, mean_val_metrics

def test(model, test_loader, device, vis_save_dir):
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for idx, (imgs, masks, edge_masks) in enumerate(tqdm(test_loader, desc="Testing")):
            imgs  = imgs.to(device)
            
            masks = masks.float().to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
                
            edge_masks_tensor = edge_masks.float().to(device) 
            if edge_masks_tensor.dim() == 3:
                edge_masks_tensor = edge_masks_tensor.unsqueeze(1)
            
            out4_logits, out3_logits, out2_logits, out1_logits = model(imgs)
            
            preds_prob = torch.sigmoid(out1_logits)
            preds = (preds_prob > 0.5).float()
            
            preds_edge_prob = torch.sigmoid(out1_logits)
            preds_edge = (preds_edge_prob > 0.5).float()

            for i in range(imgs.size(0)):
                visualize_and_save(
                    imgs[i].cpu(),
                    masks[i].squeeze(0).cpu(),
                    preds[i].squeeze(0).cpu(),
                    edge_masks_tensor[i].squeeze(0).cpu(), 
                    preds_edge[i].squeeze(0).cpu(),        
                    vis_save_dir,
                    idx * imgs.size(0) + i
                )

                m = compute_metrics_single(
                    masks[i].squeeze(0).cpu().numpy(),
                    preds[i].squeeze(0).cpu().numpy()
                )
                metrics_list.append(m)

    df = pd.DataFrame(metrics_list)
    mean_metrics = df.mean().to_dict()

    ordered_keys = ['Dice', 'IoU', 'HSD', 'MAE', 'Accuracy', 'Precision', 'Recall']
    
    print("\n=== Test Metrics (Binary Average over Images) ===")
    for k in ordered_keys:
        print(f"{k}: {mean_metrics[k]:.4f}")

    return mean_metrics

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, mode='min', delta=0.0):
        assert mode in ('min', 'max'), "mode must be 'min' or 'max'"
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.early_stop = False

    def _is_improved(self, current):
        if self.best is None:
            return True
        if self.mode == 'min':
            return current < self.best - self.delta
        else:
            return current > self.best + self.delta

    def step(self, current):
        if self._is_improved(current):
            self.best = current
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improved monitored value to {current:.6f}")
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs (current={current:.6f}, best={self.best:.6f}).")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def main(train_loader, val_loader, test_loader, epochs, lr, weight_decay, device,
         best_model_save_dir, vis_save_dir, earlystop_patience,
         monitor_metric='val_dice',  
         earlystop_delta=0.0): 

    start_time = time.time()
    model     = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max',factor=0.5,patience=5,min_lr=1e-7)

    if monitor_metric == 'val_loss':
        es_mode = 'min'
    elif monitor_metric in ('val_dice', 'val_iou'):
        es_mode = 'max'
    else:
        raise ValueError("monitor_metric must be one of 'val_loss', 'val_dice', 'val_iou'")

    early_stop = EarlyStopping(patience=earlystop_patience, verbose=True, mode=es_mode, delta=earlystop_delta)

    best_val_dice = 0.0
    best_val_iou = 0.0
    best_val_loss = float('inf')
    os.makedirs(best_model_save_dir, exist_ok=True)
    best_model_path = os.path.join(best_model_save_dir, 'best_model.pth')
    
    print(f"[Run] monitor_metric = {monitor_metric}, early_stop mode = {es_mode}, patience = {earlystop_patience}, delta = {earlystop_delta}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, scaler, device
        )
        
        val_loss, val_detailed_metrics = validate_one_epoch(
            model, val_loader, device
        )

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
        
        print(f"Val Detailed -> Acc: {val_detailed_metrics['Accuracy']:.4f} | "
              f"Prec: {val_detailed_metrics['Precision']:.4f} | "
              f"Recall: {val_detailed_metrics['Recall']:.4f} | "
              f"Dice: {val_detailed_metrics['Dice']:.4f} | "
              f"IoU: {val_detailed_metrics['IoU']:.4f} | "
              f"HSD: {val_detailed_metrics['HSD']:.4f} | "
              f"MAE: {val_detailed_metrics['MAE']:.4f}")
              
        if monitor_metric == 'val_loss':
            monitored_value = val_loss
        elif monitor_metric == 'val_dice':
            monitored_value = val_detailed_metrics['Dice']
        else:  
            monitored_value = val_detailed_metrics['IoU']

        improved = False
        if monitor_metric == 'val_loss':
            if monitored_value < best_val_loss - 1e-8:
                best_val_loss = monitored_value
                improved = True
        elif monitor_metric == 'val_dice':
            if monitored_value > best_val_dice + 1e-8:
                best_val_dice = monitored_value
                improved = True
        else:  
            if monitored_value > best_val_iou + 1e-8:
                best_val_iou = monitored_value
                improved = True

        if improved:
            torch.save(model.state_dict(), best_model_path)
            print(f"[Model Saved] {monitor_metric} improved to {monitored_value:.6f}. Model saved to {best_model_path}")
        else:
            print(f"No improvement on {monitor_metric}. Current: {monitored_value:.6f}; Best (loss/dice/iou): {best_val_loss:.6f}/{best_val_dice:.6f}/{best_val_iou:.6f}")

        scheduler.step(val_detailed_metrics['Dice'])

        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Current learning rates: {current_lrs}")

        early_stop.step(monitored_value)
        if early_stop.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement on {monitor_metric} for {early_stop.patience} epochs.")
            break
            
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal training time: {total_time:.2f}s, "
          f"Avg per epoch: {total_time/(epoch+1):.2f}s")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    os.makedirs(vis_save_dir, exist_ok=True)
    test(model, test_loader, device, vis_save_dir)

if __name__ == "__main__":   
    datasets_config = [
        
        {
            "name": "BUSI", 
            "path": "......................", 
        },
    ]
    
    batch_size = 8
    epochs = 250
    earlystop_patience = 30
    lr = 2e-4
    weight_decay = 1e-4
    device = 'cuda'
    
    weights_source_dir = "..............................." 
    test_results_base_dir = "................................" 
    is_test_mode = False

    for config in datasets_config:
        dataset_name = config["name"]
        dataset_path = config["path"]

        print(f"\n{'='*20} Loading: {dataset_name} {'='*20}")
        
        source_root = os.path.join(weights_source_dir, dataset_name)
        ckpt_path_to_load = os.path.join(source_root, "checkpoints", "best_model.pth")
        
        if is_test_mode:
            current_save_root = os.path.join(test_results_base_dir, dataset_name)
        else:
            current_save_root = source_root

        best_model_save_dir = os.path.join(current_save_root, "checkpoints")
        vis_save_dir = os.path.join(current_save_root, "visualization_results")

        os.makedirs(best_model_save_dir, exist_ok=True)
        os.makedirs(vis_save_dir, exist_ok=True)

        print(f"data path: {dataset_path}")
        print(f"ckpt path: {ckpt_path_to_load}") 
        print(f"Saved to: {current_save_root}")

        try:
            train_loader, val_loader, test_loader = get_dataloaders(dataset_path, batch_size=batch_size)
        except Exception as e:
            print(f"[Error] Load {dataset_name} Failed: {e}")
            continue 

        if is_test_mode:
            print(f"[{dataset_name}] Train...")
            model = UNet().to(device)
            if os.path.exists(ckpt_path_to_load):
                print(f"Loading weights from: {ckpt_path_to_load}")
                model.load_state_dict(torch.load(ckpt_path_to_load, weights_only=True))
                test(model, test_loader, device, vis_save_dir)
            else:
                print(f"[Warning] Not Found: {ckpt_path_to_load}")
        else:
            print(f"[{dataset_name}] Testing...")
            main(train_loader, val_loader, test_loader, epochs, lr, weight_decay, device,
                 best_model_save_dir, vis_save_dir, earlystop_patience)

        if 'model' in locals(): del model
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        if 'test_loader' in locals(): del test_loader
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"[{dataset_name}] Finished\n")

    print("#####################")