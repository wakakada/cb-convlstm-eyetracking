# 瞳孔跟踪模型训练
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
from torch.amp import autocast, GradScaler

# --- 配置 ---
HEIGHT, WIDTH = 45, 60
SEQ_LEN = 20
BATCH_SIZE = 128
NUM_EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device:{DEVICE}")


# 早停配置
EARLY_STOPPING_PATIENCE = 5         # 验证损失多少个epoch不下降就停止
EARLY_STOPPING_MIN_DELTA = 0.0001   # 最小改善阈值


# 定义瞳孔跟踪的神经网络结构
class PupilTrackerModel(nn.Module):
    def __init__(self, height, width, input_dim=1):
        super(PupilTrackerModel, self).__init__()
        # 卷积长短期记忆网络，同时提取空间特征和时间依赖
        self.convlstm1 = ConvLSTM(input_dim=input_dim, hidden_dim=16, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm3d(16)   # 对3D特征图((batch, channel, seq, h, w))进行批归一化，加速收敛
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))    # 3D最大池化，kernel_size=(1, 2, 2)，表示仅在空间维度(H, W)下采样2倍，时间维度保持不变

        self.convlstm2 = ConvLSTM(input_dim=16, hidden_dim=32, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm3 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # 动态计算全连接输入维度，适用于输入尺寸可能变化的情况，确保模型的通用性和灵活性
        self.fc1_dyn = None
        self.fc2_dyn = None

    def forward(self, x):
        x, _ = self.convlstm1(x)    # 返回一个元组(output, (h, c))
        x = x[0].permute(0, 2, 1, 3, 4) # 将通道维与时间维交换->(batch, channels, seq, h, w)，这是BatchNorm3d和MaxPool3d期望的格式
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = x.permute(0, 2, 1, 3, 4)    # ->(batch, seq, channels, h, w)
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

        B, C, T, H, W = x.size()

        outputs = []
        for t in range(T):
            # 对每个时间步t，将当前时刻的特征图展平为一维向量(B, C*H*W)
            feat = x[:, :, t, :, :].reshape(B, -1)
            if self.fc1_dyn is None:
                # 首次运行时，根据展平特征维度创建全连接层。fc1将特征映射到128维，fc2输出2个坐标（瞳孔中心的x，y）
                self.fc1_dyn = nn.Linear(feat.size(1), 128).to(DEVICE)
                self.fc2_dyn = nn.Linear(128, 2).to(DEVICE)

            feat = torch.relu(self.fc1_dyn(feat))   # 对每个时间步应用相同的全连接层（权值共享）
            feat = nn.Dropout(0.5)(feat)
            out = self.fc2_dyn(feat)
            outputs.append(out)
        # 将所有时间步的输出堆叠并调整维度为(batch, seq, 2)，对应每帧的预测坐标
        y = torch.stack(outputs, dim=0).permute(1, 0, 2)
        return y


class SmoothPupilTrackerModel(PupilTrackerModel):
    """带轨迹平滑约束的瞳孔跟踪模型，添加轨迹平滑约束，用于鼓励模型预测坐标在时间上连续变化，减少抖动"""
    def __init__(self, height, width, input_dim=1, smooth_weight=0.1):
        super(SmoothPupilTrackerModel, self).__init__(height, width, input_dim)
        self.smooth_weight = smooth_weight  # 平滑损失在总损失中的权重

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
    # lpw_root = "E:\school\毕设\convlstm-eyetracking\LPW"
    # train_list = "train_files.txt"
    # val_list = "val_files.txt"

    lpw_root = "/root/cb-convlstm-eyetracking/CloudData/LPW"          # AI galaxy
    train_list = "/root/cb-convlstm-eyetracking/eyetracking-convlstm/train_files.txt"
    val_list = "/root/cb-convlstm-eyetracking/eyetracking-convlstm/val_files.txt"

    # lpw_root = os.path.join("/kaggle/input/datasets/wakakaele/eyetracking-lpw", "LPW")
    # train_list = os.path.join("/kaggle/input/models/wakakaele/eyetracking/pytorch/default/1", "train_files.txt")
    # val_list = os.path.join("/kaggle/input/models/wakakaele/eyetracking/pytorch/default/1", "val_files.txt")

    # 创建数据集；stride：采样滑动窗口的步长
    train_dataset = LPWDataset(lpw_root, train_list, seq_len=SEQ_LEN, stride=1, img_size=(HEIGHT, WIDTH), dataset_type="train", preload=True) # 训练集stride=1以生成大量样本
    val_dataset = LPWDataset(lpw_root, val_list, seq_len=SEQ_LEN, stride=SEQ_LEN, img_size=(HEIGHT, WIDTH), dataset_type="val", preload=True) # 验证集stride=SEQ_LEN避免重叠，保证评估独立性
    # num_workers=4：多进程加载数据，加快I/O；pin_memory=True：将数据锁页在内存，加速GPU传输
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = PupilTrackerModel(HEIGHT, WIDTH).to(DEVICE)

    # 预初始化全连接层
    dummy_input = torch.randn(1, SEQ_LEN, 1, HEIGHT, WIDTH).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy_input)

    criterion = nn.SmoothL1Loss()   #Huber损失，对离群点鲁棒，适合回归任务
    optimizer = optim.Adam(model.parameters(), lr=LR)   # Adam优化器

    # 学习率调度器，当验证损失停止下降时，学习率乘以factor=0.5，patience=5表示连续5个epoch不降则调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()   # 混合精度训练的梯度缩放器，防止低精度下梯度下溢
    # 添加平滑损失权重
    smooth_weight = 0.1

    # 训练循环，记录最佳验证损失、最佳轮次、当前未改善计数
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    # 保存每轮指标，供后续绘图
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

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)
        for batch_x, batch_y in pbar:   # 遍历训练集
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # 混合精度训练
            with autocast(device_type = 'cuda'):
                outputs = model(batch_x)
                detection_loss = criterion(outputs, batch_y)

                if isinstance(model, SmoothPupilTrackerModel):
                    # 若模型带平滑损失，则额外计算并加权求和
                    smooth_loss = model.smoothness_loss(outputs)
                    total_loss_value = detection_loss + smooth_weight * smooth_loss
                    total_smooth_loss += smooth_loss.item()
                else:
                    total_loss_value = detection_loss

            # 缩放梯度反向传播
            scaler.scale(total_loss_value).backward()
            scaler.step(optimizer)  # 更新参数
            scaler.update()         # 调整缩放因子

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

        # 验证阶段不计算梯度，仅使用检测损失（不加平滑损失）
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader: # 遍历验证集
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
