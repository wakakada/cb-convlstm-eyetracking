# decord读视频加快速度
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm

# 引入 decord读视频
import decord
from decord import VideoReader
from decord import cpu, gpu
import torch.nn.functional as F

class LPWDataset(Dataset):
    def __init__(self, lpw_root, video_list_file, seq_len=40, stride=1, img_size=(60, 80), dataset_type='train', preload=False):
        """
        lpw_root: LPW 数据集根目录
        video_list_file: 视频列表文件
        seq_len: 时间序列长度
        stride: 采样步长
        img_size: (height, width) 模型输入尺寸
        dataset_type: 'train' 或 'val'
        preload: 是否预加载到内存 (建议 False，让 GPU 按需读取)
        """
        self.lpw_root = lpw_root
        self.seq_len = seq_len
        self.stride = stride
        self.img_h, self.img_w = img_size # 注意顺序是 H, W
        self.dataset_type = dataset_type
        self.preload = preload
        
        self.samples = []
        self.video_cache = {}       
        self._labels_cache = {}     
        self.video_shapes = {}      

        # 加载视频列表
        with open(os.path.join(os.path.dirname(__file__), video_list_file), 'r') as f:
            video_names = [line.strip() for line in f.readlines()]

        # 构建路径
        for name in video_names:
            parts = name.split('_')
            # 兼容 subj_X_vid_Y 格式
            if len(parts) >= 4:
                subj_id = parts[1]
                vid_id = parts[3]
            else:
                # 如果是其他格式，尝试直接分割
                # 这里假设你的文件名逻辑没变
                continue 

            video_path = os.path.join(lpw_root, subj_id, f"{vid_id}.avi")
            label_path = os.path.join(lpw_root, subj_id, f"{vid_id}.txt")
            
            if os.path.exists(video_path) and os.path.exists(label_path):
                self.samples.append((video_path, label_path))

        dataset_name = "训练集" if dataset_type == 'train' else "验证集"

        if self.preload:
            print(f"正在预加载{len(self.samples)}个视频到内存... ({dataset_name})")
            # 注意：如果用 GPU 解码，preload 可能会爆显存。
            # 建议 preload 时使用 CPU 解码 ctx=cpu(0)
            for v_idx, (v_path, l_path) in enumerate(tqdm(self.samples, desc=f"Loading {dataset_name}")):
                frames = self._load_video_frames(v_path)    
                labels = self._load_labels(l_path)          
                self.video_cache[v_path] = frames
                self._labels_cache[l_path] = labels
                if len(frames) > 0:
                    # decord 读出来的是 (H, W, C)，shape[0] 是 H
                    h, w = frames[0].shape[:2]
                    self.video_shapes[v_path] = (w, h)  
                else:
                    self.video_shapes[v_path] = (0, 0)
            print(f"✓ {dataset_name}预加载完成！")
        else:
            # 仅加载元数据 (使用 decord 快速获取宽高，比 cv2 快)
            print(f"正在加载{dataset_name}元数据...")
            for v_path, l_path in tqdm(self.samples, desc=f"Loading {dataset_name}"):
                labels = self._load_labels(l_path)
                self._labels_cache[l_path] = labels

                # 使用 decord 快速获取视频信息
                try:
                    vr = VideoReader(v_path, ctx=cpu(0))
                    self.video_shapes[v_path] = (vr.width, vr.height)
                except:
                    self.video_shapes[v_path] = (0, 0)
            print(f"✓ {dataset_name}元数据加载完成！")

        # 生成滑动窗口索引
        self.frame_indices = []
        for v_idx, (v_path, l_path) in enumerate(self.samples):
            num_frames = len(self._labels_cache[l_path])
            if num_frames < seq_len:
                continue

            for start in range(0, num_frames - seq_len + 1, stride):
                self.frame_indices.append((v_idx, start))
        print(f"✓ {dataset_name}加载完成，共 {len(self.frame_indices)} 个样本，{len(self.samples)} 个视频\n")

    def _load_video_frames(self, video_path):
        """预加载视频所有帧 (使用 decord)"""
        # 预加载建议用 CPU 解码，避免爆显存
        vr = VideoReader(video_path, ctx=cpu(0))
        indices = list(range(len(vr)))
        frames = vr.get_batch(indices) # (N, H, W, C)
        frames = frames.asnumpy() # 转 Numpy (N, H, W, C)
        
        # 处理：灰度化、缩放、归一化
        processed_frames = []
        for frame in frames:
            # 简单灰度化 (取平均或加权)
            gray = np.mean(frame, axis=2) # (H, W)
            # Resize
            resized = np.array(Image.fromarray(gray).resize((self.img_w, self.img_h))) # 需要 import Image
            # 归一化
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)
        return np.array(processed_frames)

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
        v_path, l_path = self.samples[v_idx]

        if self.preload:
            frames = self.video_cache[v_path][start_frame:start_frame + self.seq_len]
            labels = self._labels_cache[l_path][start_frame:start_frame + self.seq_len]
            data_tensor = np.expand_dims(frames, axis=1) # (Seq, 1, H, W)
            label_tensor = labels
        else:
            # --- 核心修改：使用 decord GPU 解码 ---
            # 注意：如果在 Kaggle 上报 CUDA 错误，请把 ctx=gpu(0) 改为 ctx=cpu(0)
            vr = VideoReader(v_path, ctx=cpu(0)) 
            
            # 获取指定帧索引
            indices = list(range(start_frame, start_frame + self.seq_len))
            
            # 读取帧 (直接得到 GPU Tensor: Seq, H, W, C)
            frames = vr.get_batch(indices)
            
            # 转换为 (Seq, C, H, W)
            frames = frames.permute(0, 3, 1, 2)
            
            # 归一化
            frames = frames.float() / 255.0
            
            # GPU 上 Resize (双线性插值)
            frames = F.interpolate(frames, size=(self.img_h, self.img_w), mode='bilinear', align_corners=False)
            
            # GPU 上转灰度 (取 RGB 平均值，简单有效)
            # frames shape: (Seq, 3, H, W) -> (Seq, 1, H, W)
            frames = frames.mean(dim=1, keepdim=True)
            
            # 转回 CPU 给 DataLoader (或者保持 GPU，看 DataLoader 设置)
            # 通常 DataLoader 期望 CPU 数据，但我们可以返回 GPU Tensor 并在 collate_fn 处理
            # 为了兼容默认 DataLoader，这里转回 CPU
            data_tensor = frames.cpu().numpy() # (Seq, 1, H, W)

            labels = self._labels_cache[l_path]
            labels = labels[start_frame:start_frame + self.seq_len]
            label_tensor = labels

        # 坐标归一化
        orig_w, orig_h = self.video_shapes[v_path]
        norm_labels = []
        for i in range(self.seq_len):
            lx, ly = label_tensor[i]
            # 防止除以 0
            if orig_w == 0: orig_w = 1
            if orig_h == 0: orig_h = 1
            nx = (lx / orig_w)
            ny = (ly / orig_h)
            norm_labels.append([nx, ny])

        label_tensor = np.array(norm_labels, dtype=np.float32)

        return torch.from_numpy(data_tensor), torch.from_numpy(label_tensor)
