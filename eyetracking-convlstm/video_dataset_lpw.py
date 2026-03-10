# file: video_dataset_lpw.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class LPWDataset(Dataset):
    def __init__(self, lpw_root, video_list_file, seq_len=40, stride=1, img_size=(60, 80)):
        """
        lpw_root: LPW 数据集根目录 (如 ./LPW/)
        video_list_file: 视频列表文件 (如 train_files.txt)
        seq_len: 时间序列长度
        stride: 采样步长
        img_size: (height, width) 模型输入尺寸
        """
        self.lpw_root = lpw_root
        self.seq_len = seq_len
        self.stride = stride
        self.img_size = img_size
        self.samples = []

        # 加载视频列表
        with open(video_list_file, 'r') as f:
            video_names = [line.strip() for line in f.readlines()]

        # 构建视频和标签路径 (格式：subj_X_vid_Y)
        for name in video_names:
            # LPW 格式：subj_X_vid_Y -> 参与者 X, 视频 Y
            parts = name.split('_')
            subj_id = parts[1]  # 如 "10"
            vid_id = parts[3]  # 如 "8"

            video_path = os.path.join(lpw_root, subj_id, f"{vid_id}.avi")
            label_path = os.path.join(lpw_root, subj_id, f"{vid_id}.txt")

            if os.path.exists(video_path) and os.path.exists(label_path):
                self.samples.append((video_path, label_path))

        # 生成滑动窗口索引
        self.frame_indices = []
        for v_idx, (v_path, l_path) in enumerate(self.samples):
            labels = self._load_labels(l_path)
            num_frames = len(labels)

            if num_frames < seq_len:
                continue

            for start in range(0, num_frames - seq_len + 1, stride):
                self.frame_indices.append((v_idx, start))

    def _load_labels(self, path):
        labels = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels.append([float(parts[0]), float(parts[1])])
        return labels

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        v_idx, start_frame = self.frame_indices[idx]
        v_path, l_path = self.samples[v_idx]

        # 获取原始视频分辨率
        cap_temp = cv2.VideoCapture(v_path)
        orig_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_temp.release()

        # 读取视频帧
        cap = cv2.VideoCapture(v_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        labels = self._load_labels(l_path)

        for i in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.img_size[1], self.img_size[0]))
            normalized = resized.astype(np.float32) / 255.0
            frames.append(normalized)

        cap.release()

        # 标签归一化到 [0, 1]
        norm_labels = []
        for i in range(self.seq_len):
            if start_frame + i < len(labels):
                lx, ly = labels[start_frame + i]
                # 归一化到输入尺寸比例
                nx = (lx / orig_w)
                ny = (ly / orig_h)
                norm_labels.append([nx, ny])
            else:
                norm_labels.append(norm_labels[-1])

        data_tensor = np.array(frames)
        data_tensor = np.expand_dims(data_tensor, axis=1)  # (Seq, 1, H, W)
        label_tensor = np.array(norm_labels, dtype=np.float32)  # (Seq, 2)

        return torch.from_numpy(data_tensor), torch.from_numpy(label_tensor)
