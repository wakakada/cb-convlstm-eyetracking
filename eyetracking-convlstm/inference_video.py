import cv2
import torch
import numpy as np
from train_video_lpw import PupilTrackerModel, HEIGHT, WIDTH, SEQ_LEN
import os


def run_inference(video_path, model_path, output_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = PupilTrackerModel(HEIGHT, WIDTH).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    frame_buffer = []
    frame_count = 0

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (WIDTH, HEIGHT))
        normalized = resized.astype(np.float32) / 255.0
        frame_buffer.append(normalized)

        # 当缓冲区满时进行预测
        if len(frame_buffer) == SEQ_LEN:
            # 构造输入 Tensor: (1, Seq, 1, H, W)
            input_data = np.array(frame_buffer)
            input_data = np.expand_dims(input_data, axis=0)  # Batch=1
            input_data = np.expand_dims(input_data, axis=2)  # Channel=1
            input_tensor = torch.from_numpy(input_data).to(DEVICE).float()

            with torch.no_grad():
                # 输出形状: (1, Seq, 2)
                prediction = model(input_tensor)

                # 取最后一个时间步的预测作为当前帧的结果 (或者取平均)
                # prediction[0, -1] 对应的是 buffer 中最后一帧的坐标 (归一化 0-1)
                pred_norm = prediction[0, -1].cpu().numpy()

                # 还原到原始视频分辨率
                pred_x = int(pred_norm[0] * WIDTH * (orig_w / WIDTH))
                pred_y = int(pred_norm[1] * HEIGHT * (orig_h / HEIGHT))

                # 实际上上面的换算逻辑有点绕，简单点：
                # 模型输出是相对于 (WIDTH, HEIGHT) 的比例。
                # 真实坐标 = 比例 * 原始宽/高
                real_x = int(pred_norm[0] * orig_w)
                real_y = int(pred_norm[1] * orig_h)

                # 绘制圆圈
                cv2.circle(frame, (real_x, real_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (real_x, real_y), 10, (255, 0, 0), 1)

                # 显示坐标
                cv2.putText(frame, f"Pupil: ({real_x}, {real_y})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 移除第一帧，保持滑动窗口
            frame_buffer.pop(0)

            out.write(frame)
            frame_count += 1

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    # 使用方法
    # python inference_video.py
    video_src = "test_input.mp4"  # 替换为你的视频
    model_ckpt = "pupil_tracker_lpw.pth"
    video_dst = "test_output_labeled.mp4"

    if os.path.exists(video_src):
        run_inference(video_src, model_ckpt, video_dst)
    else:
        print(f"Video file {video_src} not found.")
