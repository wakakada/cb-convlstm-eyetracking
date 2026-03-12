import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from video_dataset_lpw import LPWDataset
from convlstm_delta import ConvLSTM
import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# --- 配置 ---
HEIGHT, WIDTH = 45, 60
SEQ_LEN = 40
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 早停配置
EARLY_STOPPING_PATIENCE = 5         # 验证损失多少个epoch不下降就停止
EARLY_STOPPING_MIN_DELTA = 0.0001   # 最小改善阈值


# --- 模型定义 ---
class PupilTrackerModel(nn.Module):
    def __init__(self, height, width, input_dim=1):
        super(PupilTrackerModel, self).__init__()
        self.convlstm1 = ConvLSTM(input_dim=input_dim, hidden_dim=16, kernel_size=(3, 3), num_layers=1,
                                  batch_first=True)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm2 = ConvLSTM(input_dim=16, hidden_dim=32, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm3 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm4 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # 动态计算全连接输入维度
        self.fc1_dyn = None
        self.fc2_dyn = None

    def forward(self, x):
        x, _ = self.convlstm1(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm2(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm3(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm4(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)

        B, C, T, H, W = x.size()

        outputs = []
        for t in range(T):
            feat = x[:, :, t, :, :].reshape(B, -1)

            if self.fc1_dyn is None:
                self.fc1_dyn = nn.Linear(feat.size(1), 128).to(DEVICE)
                self.fc2_dyn = nn.Linear(128, 2).to(DEVICE)

            feat = torch.relu(self.fc1_dyn(feat))
            feat = nn.Dropout(0.5)(feat)
            out = self.fc2_dyn(feat)
            outputs.append(out)

        y = torch.stack(outputs, dim=0).permute(1, 0, 2)
        return y


class SmoothPupilTrackerModel(PupilTrackerModel):
    """带轨迹平滑约束的瞳孔跟踪模型"""

    def __init__(self, height, width, input_dim=1, smooth_weight=0.1):
        super(SmoothPupilTrackerModel, self).__init__(height, width, input_dim)
        self.smooth_weight = smooth_weight

    def smoothness_loss(self, outputs):
        """
        计算轨迹平滑性损失
        惩罚相邻帧之间的突变
        """
        if outputs.size(1) < 3:
            return torch.tensor(0.0, device=outputs.device)

        # 计算一阶差分 (速度)
        diff1 = outputs[:, 1:, :] - outputs[:, :-1, :]

        # 计算二阶差分 (加速度)
        diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]

        # 惩罚加速度
        smooth_loss = torch.mean(diff2 ** 2)

        return smooth_loss


if __name__ == "__main__":
    # LPW 数据集路径
    # lpw_root = "/root/autodl-tmp/LPW/"    # autodl
    lpw_root = "E:\school\毕设\convlstm-eyetracking\LPW"
    train_list = "train_files.txt"
    val_list = "val_files.txt"

    # lpw_root = "/root/cb-convlstm-eyetracking/CloudData/LPW"          # AI galaxy
    # train_list = "/root/cb-convlstm-eyetracking/eyetracking-convlstm/train_files.txt"
    # val_list = "/root/cb-convlstm-eyetracking/eyetracking-convlstm/val_files.txt"

    # 创建数据集
    train_dataset = LPWDataset(lpw_root, train_list, seq_len=SEQ_LEN, stride=1, img_size=(HEIGHT, WIDTH), dataset_type="train")
    val_dataset = LPWDataset(lpw_root, val_list, seq_len=SEQ_LEN, stride=SEQ_LEN, img_size=(HEIGHT, WIDTH), dataset_type="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化模型
    model = PupilTrackerModel(HEIGHT, WIDTH).to(DEVICE)

    # 预初始化全连接层
    dummy_input = torch.randn(1, SEQ_LEN, 1, HEIGHT, WIDTH).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy_input)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()
    # 添加平滑损失权重
    smooth_weight = 0.1

    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    print("Start Training with LPW Dataset...")
    print(f"Early Stopping: patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}")
    print("="*60)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_smooth_loss = 0

        # 替换原来的训练循环内部
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # 混合精度训练
            with autocast():
                outputs = model(batch_x)
                detection_loss = criterion(outputs, batch_y)

                if isinstance(model, SmoothPupilTrackerModel):
                    smooth_loss = model.smoothness_loss(outputs)
                    total_loss_value = detection_loss + smooth_weight * smooth_loss
                    total_smooth_loss += smooth_loss.item()
                else:
                    total_loss_value = detection_loss

            # 缩放梯度反向传播
            scaler.scale(total_loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += detection_loss.item()
            # 更新进度条
            pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}'})

        epoch_loss = total_loss / len(train_loader)
        avg_smooth_loss = total_smooth_loss / len(train_loader)
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {epoch_loss:.6f}, Smooth Loss: {avg_smooth_loss:.6f}")

        # 记录训练历史
        history['train_loss'].append(epoch_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE).float()
                batch_y = batch_y.to(DEVICE).float()
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # 打印当前epoch结果
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # 学习率调整
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr < old_lr:
            print(f" Learning rate adjusted: {old_lr:.6f} -> {new_lr:.6f}")

        # 保存最佳模型
        improved = val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, "pupil_tracker_lpw.pth")
            print(f"Saved best model (improvement: {best_val_loss:.6f})!")
        else:
            patience_counter += 1
            print(f"No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")

        print("="*60)

        # 早停检查
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n Early stopping triggered at epoch {epoch + 1}!")
            print(f"   Best model was saved at epoch {best_epoch} with val_loss={best_val_loss:.6f}")
            break

    print("\n" + "=" * 60)
    print("Training Finished.")
    print(f"Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}")
    print("=" * 60)

    # 绘制损失函数图像
    plt.figure(figsize=(12, 5))

    # 子图 1: 训练损失和验证损失
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 2: 学习率变化
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['lr'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
    print("\n✓ Loss curves saved to 'training_loss_curve.png'")

    # 保存训练历史
    np.save('training_history.npy', history)
    print("✓ Training history saved to 'training_history.npy'")
