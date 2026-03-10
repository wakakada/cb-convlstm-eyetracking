import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from video_dataset_lpw import LPWDataset
from convlstm_delta import ConvLSTM
import tqdm
import os

# --- 配置 ---
HEIGHT, WIDTH = 60, 80
SEQ_LEN = 40
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# --- 主程序 ---
if __name__ == "__main__":
    # LPW 数据集路径
    lpw_root = "../LPW/"  # 根据实际路径调整
    train_list = "train_files.txt"
    val_list = "val_files.txt"

    # 创建数据集
    train_dataset = LPWDataset(lpw_root, train_list, seq_len=SEQ_LEN, stride=1, img_size=(HEIGHT, WIDTH))
    val_dataset = LPWDataset(lpw_root, val_list, seq_len=SEQ_LEN, stride=SEQ_LEN, img_size=(HEIGHT, WIDTH))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型
    model = PupilTrackerModel(HEIGHT, WIDTH).to(DEVICE)

    # 预初始化全连接层
    dummy_input = torch.randn(1, SEQ_LEN, 1, HEIGHT, WIDTH).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy_input)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    best_val_loss = float('inf')
    print("Start Training with LPW Dataset...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in tqdm.tqdm(train_loader):
            batch_x = batch_x.to(DEVICE).float()
            batch_y = batch_y.to(DEVICE).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}")

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
        print(f"Validation Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, "pupil_tracker_lpw.pth")
            print("Saved best model!")

    print("Training Finished.")
