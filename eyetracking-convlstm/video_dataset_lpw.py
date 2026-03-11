import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class LPWDataset(Dataset):
    def __init__(self, lpw_root, video_list_file, seq_len=40, stride=1, img_size=(60, 80), dataset_type='train'):
        """
        lpw_root: LPW 数据集根目录 (如 ./LPW/)
        video_list_file: 视频列表文件 (如 train_files.txt)
        seq_len: 时间序列长度
        stride: 采样步长
        img_size: (height, width) 模型输入尺寸
        dataset_type:数据集类型 ('train' 或 'val')
        """
        self.lpw_root = lpw_root
        self.seq_len = seq_len
        self.stride = stride
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.samples = []
        # 预先加载所有视频到内存，避免反复读取文件
        self.video_cache = []
        self._labels_cache = []
        self.video_shapes = []      # 存储原始视频分辨率

        # 加载视频列表
        with open(video_list_file, 'r') as f:
            video_names = [line.strip() for line in f.readlines()]

        # 构建视频和标签路径 (格式：subj_X_vid_Y)
        for name in video_names:
            # LPW 格式：subj_X_vid_Y -> 参与者 X, 视频 Y
            parts = name.split('_')
            subj_id = parts[1]
            vid_id = parts[3]

            video_path = os.path.join(lpw_root, subj_id, f"{vid_id}.avi")
            label_path = os.path.join(lpw_root, subj_id, f"{vid_id}.txt")

            if os.path.exists(video_path) and os.path.exists(label_path):
                self.samples.append((video_path, label_path))

        # 预先加载所有视频到内存
        dataset_name = "训练集" if dataset_type == 'train' else "验证集"
        print(f"正在预加载{len(self.samples)}个视频到内存...")
        for v_idx, (v_path, l_path) in enumerate(self.samples):
            frames = self._load_video_frames(v_path)
            labels = self._load_labels(l_path)
            self.video_cache.append(frames)
            self._labels_cache.append(labels)
            # 保存原始视频分辨率
            if len(frames) > 0:
                h, w = frames[0].shape
                self.video_shapes.append((w, h))  # (width, height)
            else:
                self.video_shapes.append((0, 0))

            if (v_idx + 1) % 5 == 0:
                print(f"已加载{v_idx + 1}/{len(self.samples)}个视频")
        print("视频预加载完成！")

        # 生成滑动窗口索引
        self.frame_indices = []
        for v_idx in range(len(self.samples)):
            num_frames = len(self._labels_cache[v_idx])

            if num_frames < seq_len:
                continue

            for start in range(0, num_frames - seq_len + 1, stride):
                self.frame_indices.append((v_idx, start))
        print(f"✓ {dataset_name}加载完成，共 {len(self.frame_indices)} 个样本，{len(self.samples)} 个视频\n")

    def _load_video_frames(self, video_path):
        """预加载视频所有帧到内存"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.img_size[1], self.img_size[0]))
            normalized = resized.astype(np.float32) / 255.0
            frames.append(normalized)
        cap.release()
        return np.array(frames)

    def _load_labels(self, path):
        labels = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels.append([float(parts[0]), float(parts[1])])
        return np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        v_idx, start_frame = self.frame_indices[idx]

        # 直接从缓存读取视频帧和标签
        frames = self.video_cache[v_idx][start_frame:start_frame + self.seq_len]
        labels = self._labels_cache[v_idx][start_frame:start_frame + self.seq_len]

        # 获取原始视频分辨率
        orig_w, orig_h = self.video_shapes[v_idx]

        # 标签归一化到 [0, 1]
        norm_labels = []
        for i in range(self.seq_len):
            lx, ly = labels[i]
            # 归一化到输入尺寸比例
            nx = (lx / orig_w)
            ny = (ly / orig_h)
            norm_labels.append([nx, ny])

        # 转换为tensor
        data_tensor = np.array(frames)
        data_tensor = np.expand_dims(data_tensor, axis=1)  # (Seq, 1, H, W)
        label_tensor = np.array(norm_labels, dtype=np.float32)  # (Seq, 2)

        return torch.from_numpy(data_tensor), torch.from_numpy(label_tensor)
