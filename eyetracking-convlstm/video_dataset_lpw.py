# decord读视频加快速度
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm

# 引入 decord
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
        with open(video_list_file, 'r') as f:
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



# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import os
# from tqdm import tqdm


# class LPWDataset(Dataset):
#     def __init__(self, lpw_root, video_list_file, seq_len=40, stride=1, img_size=(60, 80), dataset_type='train', preload=False):
#         """
#         lpw_root: LPW 数据集根目录 (如 ./LPW/)
#         video_list_file: 视频列表文件 (如 train_files.txt)
#         seq_len: 时间序列长度
#         stride: 采样步长
#         img_size: (height, width) 模型输入尺寸
#         dataset_type:数据集类型 ('train' 或 'val')
#         """
#         self.lpw_root = lpw_root
#         self.seq_len = seq_len
#         self.stride = stride
#         self.img_size = img_size
#         self.dataset_type = dataset_type
#         self.samples = []
#         self.preload = preload
#         self.video_cache = {}       # 当preload=True时，以video_path为键，存储该视频所有者帧的预处理数组
#         self._labels_cache = {}     # 以label_path为键，存储该视频所有帧的瞳孔坐标（未归一化）
#         self.video_shapes = {}      # 以video_path为键，存储原始视频分辨率（width, height）

#         # 加载视频列表
#         with open(video_list_file, 'r') as f:
#             video_names = [line.strip() for line in f.readlines()]

#         # 构建视频和标签路径 (格式：subj_X_vid_Y)
#         for name in video_names:
#             # LPW 格式：subj_X_vid_Y -> 参与者 X, 视频 Y
#             parts = name.split('_')
#             subj_id = parts[1]
#             vid_id = parts[3]

#             video_path = os.path.join(lpw_root, subj_id, f"{vid_id}.avi")
#             label_path = os.path.join(lpw_root, subj_id, f"{vid_id}.txt")
#             # 只保留视频文件和标签文件同时存在的样本
#             if os.path.exists(video_path) and os.path.exists(label_path):
#                 self.samples.append((video_path, label_path))

#         # 预先加载视频到内存
#         dataset_name = "训练集" if dataset_type == 'train' else "验证集"

#         if self.preload:
#             print(f"正在预加载{len(self.samples)}个视频到内存... ({dataset_name})")
#             for v_idx, (v_path, l_path) in enumerate(tqdm(self.samples, desc=f"Loading {dataset_name}")):
#                 frames = self._load_video_frames(v_path)    # 将全部帧读入内存并预处理
#                 labels = self._load_labels(l_path)          # 加载标签
#                 # 全部数据存入元组，后续__getitem__直接切片，避免反复读盘
#                 self.video_cache[v_path] = frames
#                 self._labels_cache[l_path] = labels
#                 if len(frames) > 0:
#                     h, w = frames[0].shape
#                     self.video_shapes[v_path] = (w, h)  # 保存原始分辨率用于坐标归一化
#                 else:
#                     self.video_shapes[v_path] = (0, 0)
#             print(f"✓ {dataset_name}预加载完成！")
#         else:
#             # 仅加载标签和视频第一帧以获取原始分辨率信息
#             print(f"正在加载{dataset_name}元数据...")
#             for v_path, l_path in tqdm(self.samples, desc=f"Loading {dataset_name}"):
#                 labels = self._load_labels(l_path)
#                 self._labels_cache[l_path] = labels

#                 cap = cv2.VideoCapture(v_path)
#                 ret, frame = cap.read()
#                 if ret:
#                     h, w = frame.shape[:2]
#                     self.video_shapes[v_path] = (w, h)
#                 else:
#                     self.video_shapes[v_path] = (0, 0)
#                 cap.release()
#             print(f"✓ {dataset_name}元数据加载完成！")

#         # 生成滑动窗口索引
#         self.frame_indices = []
#         for v_idx, (v_path, l_path) in enumerate(self.samples):
#             num_frames = len(self._labels_cache[l_path])

#             if num_frames < seq_len:
#                 continue

#             for start in range(0, num_frames - seq_len + 1, stride):    # 以stride为步长，生成起始帧索引start，每个（视频索引，起始帧）对于一个训练/验证样本
#                 self.frame_indices.append((v_idx, start))   # self.frame_indices长度即为数据集总样本数
#         print(f"✓ {dataset_name}加载完成，共 {len(self.frame_indices)} 个样本，{len(self.samples)} 个视频\n")

#     def _load_video_frames(self, video_path):
#         """预加载视频所有帧到内存"""
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # 灰度化、缩放、归一化
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             resized = cv2.resize(gray, (self.img_size[1], self.img_size[0]))
#             normalized = resized.astype(np.float32) / 255.0
#             frames.append(normalized)
#         cap.release()
#         return np.array(frames) # 返回Numpy数组(总帧数, H, W)

#     def _load_labels(self, path):
#         labels = []
#         with open(path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 2:
#                     labels.append([float(parts[0]), float(parts[1])])
#         return np.array(labels, dtype=np.float32)   # (总帧数, 2)

#     def __len__(self):
#         # 返回样本总数，供DataLoader确认迭代长度
#         return len(self.frame_indices)

#     def __getitem__(self, idx):
#         v_idx, start_frame = self.frame_indices[idx]
#         v_path, l_path = self.samples[v_idx]

#         if self.preload:
#             # 从缓存读取
#             frames = self.video_cache[v_path][start_frame:start_frame + self.seq_len]
#             labels = self._labels_cache[l_path][start_frame:start_frame + self.seq_len]
#         else:
#             # 按需加载：只读取需要的帧
#             cap = cv2.VideoCapture(v_path)
#             frames = []
#             for i in range(start_frame, start_frame + self.seq_len):
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, i)     # 定位到指定帧
#                 ret, frame = cap.read()
#                 if ret:
#                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     resized = cv2.resize(gray, (self.img_size[1], self.img_size[0]))
#                     normalized = resized.astype(np.float32) / 255.0
#                     frames.append(normalized)
#             cap.release()
#             frames = np.array(frames)

#             labels = self._labels_cache[l_path]
#             labels = labels[start_frame:start_frame + self.seq_len]

#         orig_w, orig_h = self.video_shapes[v_path]

#         norm_labels = []
#         for i in range(self.seq_len):
#             lx, ly = labels[i]
#             nx = (lx / orig_w)
#             ny = (ly / orig_h)
#             norm_labels.append([nx, ny])

#         data_tensor = np.array(frames)
#         data_tensor = np.expand_dims(data_tensor, axis=1)
#         label_tensor = np.array(norm_labels, dtype=np.float32)

#         return torch.from_numpy(data_tensor), torch.from_numpy(label_tensor)


# 将视频帧转为图片之后再读图片，速度更快
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import os
# import json
# from PIL import Image
# from tqdm import tqdm

# class LPWDataset(Dataset):
#     def __init__(self, lpw_root, video_list_file, seq_len=20, stride=1, img_size=(45, 60), dataset_type='train', meta_json_path=None):
#         """
#         lpw_root: 预处理后的图片根目录 (如 /kaggle/input/your-dataset-name/)
#         video_list_file: 视频列表文件
#         seq_len: 序列长度
#         stride: 步长
#         img_size: (height, width) 注意这里顺序是 H, W
#         meta_json_path: meta_info.json 的路径
#         """
#         self.lpw_root = lpw_root
#         self.seq_len = seq_len
#         self.stride = stride
#         # 确保 img_size 是 (H, W)
#         self.img_h, self.img_w = img_size 
#         self.dataset_type = dataset_type
#         self.samples = []
        
#         # 1. 加载元数据 (原始视频分辨率)
#         if meta_json_path and os.path.exists(meta_json_path):
#             with open(meta_json_path, 'r') as f:
#                 self.meta_info = json.load(f)
#             print(f"✓ Loaded meta info for {len(self.meta_info)} videos.")
#         else:
#             print("⚠️ Warning: meta_json_path not found. Label normalization might fail.")
#             self.meta_info = {}

#         # 2. 加载视频列表
#         with open(video_list_file, 'r') as f:
#             video_names = [line.strip() for line in f.readlines()]

#         # 3. 构建样本 (图片文件夹路径, 标签路径)
#         # 假设 train_files.txt 里的格式是 "subj_1_vid_1"
#         # 而我们的文件夹结构是 "subj_1/1"
#         # 这里做一个简单的路径映射
#         for name in video_names:
#             # 解析 subj_X_vid_Y -> subj_X/vid_Y
#             if '_vid_' in name:
#                 parts = name.split('_vid_')
#                 subj_dir = parts[0]
#                 vid_dir = parts[1]
#                 # 图片文件夹路径
#                 img_folder_path = os.path.join(lpw_root, subj_dir, vid_dir)
#             else:
#                 # 兼容其他格式，或者直接用 name
#                 img_folder_path = os.path.join(lpw_root, name)

#             # 标签路径 (假设标签文件和原始视频在同一层级，或者你需要调整这里)
#             # 如果标签文件也在 lpw_root 下，结构可能需要调整。
#             # 这里假设标签文件路径需要根据原始结构拼接，或者你已经把标签也放到了 lpw_root
#             # 为了简化，这里假设 label_path 是基于 lpw_root 寻找 .txt
#             # 注意：如果你的标签文件不在预处理目录里，你需要传入原始的 label_root
#             # 这里暂时沿用你原来的逻辑寻找 label，但你需要确保路径正确
            
#             # 修正：通常 LPW 的 label 是 subj_X/vid_Y.txt
#             label_path = os.path.join(lpw_root, subj_dir, f"{vid_dir}.txt") 
            
#             # 检查图片和标签是否存在
#             if os.path.isdir(img_folder_path) and os.path.exists(label_path):
#                 self.samples.append((img_folder_path, label_path, name))

#         # 4. 预加载标签和生成索引
#         self._labels_cache = {}
#         self.frame_indices = []
        
#         dataset_name = "训练集" if dataset_type == 'train' else "验证集"
#         print(f"正在加载{dataset_name}数据...")

#         for v_idx, (img_folder, l_path, v_name) in enumerate(tqdm(self.samples, desc=f"Loading {dataset_name}")):
#             # 加载 Label
#             labels = self._load_labels(l_path)
#             self._labels_cache[l_path] = labels
            
#             # 获取原始尺寸用于归一化
#             # key 需要和 meta_info 里的一致
#             meta_key = f"{img_folder.split('/')[-2]}/{img_folder.split('/')[-1]}" # subj_1/1
#             # 如果 meta_info 的 key 格式不一样，这里需要调整匹配逻辑
#             # 假设 meta_info 的 key 是 "subj_1/1"
            
#             if v_name in self.meta_info:
#                  self.meta_info[l_path] = self.meta_info[v_name] # 缓存到 label path 方便读取
            
#             num_frames = len(labels)
#             if num_frames < seq_len:
#                 continue

#             for start in range(0, num_frames - seq_len + 1, stride):
#                 self.frame_indices.append((v_idx, start))
        
#         print(f"✓ {dataset_name}加载完成，共 {len(self.frame_indices)} 个样本")

#     def _load_labels(self, path):
#         labels = []
#         with open(path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 2:
#                     labels.append([float(parts[0]), float(parts[1])])
#         return np.array(labels, dtype=np.float32)

#     def __len__(self):
#         return len(self.frame_indices)

#     def __getitem__(self, idx):
#         v_idx, start_frame = self.frame_indices[idx]
#         img_folder, l_path, v_name = self.samples[v_idx]
        
#         # 1. 读取图片序列 (直接读文件，极快)
#         frames = []
#         for i in range(start_frame, start_frame + self.seq_len):
#             # 构造图片路径: /kaggle/.../subj_1/1/0005.jpg
#             img_path = os.path.join(img_folder, f"{i:04d}.jpg")
            
#             # 使用 cv2 读取灰度图
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 # 容错处理
#                 img = np.zeros((self.img_h, self.img_w), dtype=np.float32)
#             else:
#                 img = img.astype(np.float32) / 255.0
            
#             frames.append(img)
        
#         # 2. 获取 Label 并归一化
#         labels = self._labels_cache[l_path][start_frame:start_frame + self.seq_len]
        
#         # 获取原始宽高
#         # 这里需要确保能取到 meta_info。如果之前缓存失败，这里会报错。
#         # 简单起见，尝试从 meta_info 找，找不到则用 img_size (这会导致坐标错误，所以必须找到)
#         orig_w, orig_h = 640, 480 # 默认值
        
#         # 尝试多种 key 匹配
#         if v_name in self.meta_info:
#             orig_w, orig_h = self.meta_info[v_name]
#         else:
#             # 尝试构造 key: subj_1/1
#              parts = v_name.split('_vid_')
#              if len(parts) == 2:
#                  key = f"{parts[0]}/{parts[1]}"
#                  if key in self.meta_info:
#                      orig_w, orig_h = self.meta_info[key]

#         # 归一化 Label
#         norm_labels = []
#         for i in range(self.seq_len):
#             lx, ly = labels[i]
#             nx = (lx / orig_w)
#             ny = (ly / orig_h)
#             norm_labels.append([nx, ny])
            
#         data_tensor = np.array(frames) # (Seq, H, W)
#         data_tensor = np.expand_dims(data_tensor, axis=1) # (Seq, 1, H, W)
#         label_tensor = np.array(norm_labels, dtype=np.float32)

#         return torch.from_numpy(data_tensor), torch.from_numpy(label_tensor)
