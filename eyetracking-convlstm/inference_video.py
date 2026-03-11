import cv2
import torch
import numpy as np
from train_video_lpw import PupilTrackerModel, HEIGHT, WIDTH, SEQ_LEN
import os
from kalman_tracker import AdaptiveKalmanTracker


def run_inference(video_path, model_path, output_path, use_kalman=True):
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

    # 初始化卡尔曼跟踪器
    kalman_tracker = AdaptiveKalmanTracker(process_noise=0.1, measurement_noise=1.0) if use_kalman else None

    # 统计信息
    stats = {
        'total_frames': 0,
        'detected_frames': 0,
        'tracked_frames': 0,
        'lost_frames': 0
    }

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

                # 取最后一个时间步的预测作为当前帧的结果
                detection_norm = prediction[0, -1].cpu().numpy()

                # 还原到原始视频分辨率
                detection_x = detection_norm[0] * orig_w
                detection_y = detection_norm[1] * orig_h
                detection = (detection_x, detection_y)

                # 计算检测置信度（基于预测的稳定性）
                if frame_count > SEQ_LEN:
                    # 简单策略：使用连续帧间差异估计置信度
                    confidence = 0.8    # 默认较高置信度
                else:
                    confidence = 0.5    # 初始阶段置信度较低

            # 使用卡尔曼滤波进行跟踪融合
            if use_kalman and kalman_tracker is not None:
                tracked_pos, status = kalman_tracker.update(detection, confidence)
                if tracked_pos is not None:
                    final_x, final_y = tracked_pos
                    stats['tracked_frames'] += 1
                    # 根据跟踪状态调整显示颜色
                    if status == 'tracking':
                        color = (0, 255, 0)     # 绿色-正常跟踪
                    elif status == 'recovered':
                        color = (255, 255, 0)    # 青色-重新捕获
                    else:
                        color = (0, 0, 255)     # 红色-跟踪丢失
                else:
                    # 没有跟踪结果，使用模型输出
                    final_x, final_y = detection_x, detection_y
                    color = (0, 255, 0)
                    stats['detected_frames'] += 1

                    # 绘制结果
                    final_x, final_y = int(final_x), int(final_y)
                    cv2.circle(frame, (final_x, final_y), 5, color, -1)
                    cv2.circle(frame, (final_x, final_y), 10, color, 1)

                    # 显示坐标和状态
                    if use_kalman and kalman_tracker is not None:
                        state = kalman_tracker.get_state()
                        if state:
                            status_text = f"Status: {state['tracking']}"
                            vel = np.sqrt(state['velocity'][0] ** 2 + state['velocity'][1] ** 2)
                            cv2.putText(frame, f"Vel: {vel:.2f}", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        else:
                            status_text = "Status: Initializing"

                        cv2.putText(frame, status_text, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv2.putText(frame, f"Pupil: ({final_x}, {final_y})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # 移除第一帧，保持滑动窗口
                    frame_buffer.pop(0)

                    out.write(frame)
                    frame_count += 1
                    stats['total_frames'] += 1

                    if frame_count % 50 == 0:
                        print(f"Processed {frame_count} frames...")

                cap.release()
                out.release()

                # 打印统计信息
                print(f"\n=== 跟踪统计 ===")
                print(f"总帧数：{stats['total_frames']}")
                print(f"检测帧数：{stats['detected_frames']}")
                print(f"跟踪帧数：{stats['tracked_frames']}")
                print(f"丢失帧数：{stats['lost_frames']}")
                if stats['total_frames'] > 0:
                    print(f"跟踪成功率：{stats['tracked_frames'] / stats['total_frames'] * 100:.2f}%")

                print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    # 使用方法
    # python inference_video.py
    video_src = "test_input.mp4"  # 替换为你的视频
    model_ckpt = "pupil_tracker_lpw.pth"
    video_dst = "test_output_labeled.mp4"

    if os.path.exists(video_src):
        run_inference(video_src, model_ckpt, video_dst, use_kalman=True)
    else:
        print(f"Video file {video_src} not found.")
