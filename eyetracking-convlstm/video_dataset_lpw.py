import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm


class LPWDataset(Dataset):
    def __init__(self, lpw_root, video_list_file, seq_len=40, stride=1, img_size=(60, 80), dataset_type='train', preload=False):
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
        self.preload = preload
        self.video_cache = {}       # 当preload=True时，以video_path为键，存储该视频所有者帧的预处理数组
        self._labels_cache = {}     # 以label_path为键，存储该视频所有帧的瞳孔坐标（未归一化）
        self.video_shapes = {}      # 以video_path为键，存储原始视频分辨率（width, height）

        # 加载视频列表
        with open(os.path.join(os.path.dirname(__file__), video_list_file), 'r') as f:
            video_names = [line.strip() for line in f.readlines()]

        # 构建视频和标签路径 (格式：subj_X_vid_Y)
        for name in video_names:
            # LPW 格式：subj_X_vid_Y -> 参与者 X, 视频 Y
            parts = name.split('_')
            subj_id = parts[1]
            vid_id = parts[3]

            video_path = os.path.join(lpw_root, subj_id, f"{vid_id}.avi")
            label_path = os.path.join(lpw_root, subj_id, f"{vid_id}.txt")
            # 只保留视频文件和标签文件同时存在的样本
            if os.path.exists(video_path) and os.path.exists(label_path):
                self.samples.append((video_path, label_path))

        # 预先加载视频到内存
        dataset_name = "训练集" if dataset_type == 'train' else "验证集"

        if self.preload:
            print(f"正在预加载{len(self.samples)}个视频到内存... ({dataset_name})")
            for v_idx, (v_path, l_path) in enumerate(tqdm(self.samples, desc=f"Loading {dataset_name}")):
                frames = self._load_video_frames(v_path)    # 将全部帧读入内存并预处理
                labels = self._load_labels(l_path)          # 加载标签
                # 全部数据存入元组，后续__getitem__直接切片，避免反复读盘
                self.video_cache[v_path] = frames
                self._labels_cache[l_path] = labels
                if len(frames) > 0:
                    h, w = frames[0].shape
                    self.video_shapes[v_path] = (w, h)  # 保存原始分辨率用于坐标归一化
                else:
                    self.video_shapes[v_path] = (0, 0)
            print(f"✓ {dataset_name}预加载完成！")
        else:
            # 仅加载标签和视频第一帧以获取原始分辨率信息
            print(f"正在加载{dataset_name}元数据...")
            for v_path, l_path in tqdm(self.samples, desc=f"Loading {dataset_name}"):
                labels = self._load_labels(l_path)
                self._labels_cache[l_path] = labels

                cap = cv2.VideoCapture(v_path)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    self.video_shapes[v_path] = (w, h)
                else:
                    self.video_shapes[v_path] = (0, 0)
                cap.release()
            print(f"✓ {dataset_name}元数据加载完成！")

        # 生成滑动窗口索引
        self.frame_indices = []
        for v_idx, (v_path, l_path) in enumerate(self.samples):
            num_frames = len(self._labels_cache[l_path])

            if num_frames < seq_len:
                continue

            for start in range(0, num_frames - seq_len + 1, stride):    # 以stride为步长，生成起始帧索引start，每个（视频索引，起始帧）对于一个训练/验证样本
                self.frame_indices.append((v_idx, start))   # self.frame_indices长度即为数据集总样本数
        print(f"✓ {dataset_name}加载完成，共 {len(self.frame_indices)} 个样本，{len(self.samples)} 个视频\n")

    def _load_video_frames(self, video_path):
        """预加载视频所有帧到内存"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 灰度化、缩放、归一化
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.img_size[1], self.img_size[0]))
            normalized = resized.astype(np.float32) / 255.0
            frames.append(normalized)
        cap.release()
        return np.array(frames) # 返回Numpy数组(总帧数, H, W)

    def _load_labels(self, path):
        labels = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels.append([float(parts[0]), float(parts[1])])
        return np.array(labels, dtype=np.float32)   # (总帧数, 2)

    def __len__(self):
        # 返回样本总数，供DataLoader确认迭代长度
        return len(self.frame_indices)

    def __getitem__(self, idx):
        v_idx, start_frame = self.frame_indices[idx]
        v_path, l_path = self.samples[v_idx]

        if self.preload:
            # 从缓存读取
            frames = self.video_cache[v_path][start_frame:start_frame + self.seq_len]
            labels = self._labels_cache[l_path][start_frame:start_frame + self.seq_len]
        else:
            # 按需加载：只读取需要的帧
            cap = cv2.VideoCapture(v_path)
            frames = []
            for i in range(start_frame, start_frame + self.seq_len):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)     # 定位到指定帧
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (self.img_size[1], self.img_size[0]))
                    normalized = resized.astype(np.float32) / 255.0
                    frames.append(normalized)
            cap.release()
            frames = np.array(frames)

            labels = self._labels_cache[l_path]
            labels = labels[start_frame:start_frame + self.seq_len]

        orig_w, orig_h = self.video_shapes[v_path]

        norm_labels = []
        for i in range(self.seq_len):
            lx, ly = labels[i]
            nx = (lx / orig_w)
            ny = (ly / orig_h)
            norm_labels.append([nx, ny])

        data_tensor = np.array(frames)
        data_tensor = np.expand_dims(data_tensor, axis=1)
        label_tensor = np.array(norm_labels, dtype=np.float32)

        return torch.from_numpy(data_tensor), torch.from_numpy(label_tensor)