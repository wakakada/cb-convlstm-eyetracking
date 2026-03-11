# NEW_FILE_CODE
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import torch
import numpy as np
from train_video_lpw import PupilTrackerModel, HEIGHT, WIDTH, SEQ_LEN
from kalman_tracker import AdaptiveKalmanTracker
import uuid
from datetime import datetime

app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 最大 500MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}

# 创建文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 加载模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "pupil_tracker_lpw.pth"


def load_model():
    """加载训练好的模型"""
    model = PupilTrackerModel(HEIGHT, WIDTH).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully on {DEVICE}")
    return model


model = load_model()


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video(video_path, output_path, use_kalman=True):
    """
    处理视频，进行瞳孔跟踪标注

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        use_kalman: 是否使用卡尔曼滤波

    Returns:
        dict: 处理统计信息
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        'processed_frames': 0,
        'tracked_frames': 0,
        'lost_frames': 0
    }

    print(f"Processing video: {video_path}")
    print(f"Resolution: {orig_w}x{orig_h}, FPS: {fps}, Total frames: {total_frames}")

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
                # 输出形状：(1, Seq, 2)
                prediction = model(input_tensor)

                # 取最后一个时间步的预测作为当前帧的检测结果
                detection_norm = prediction[0, -1].cpu().numpy()

                # 还原到原始视频分辨率
                detection_x = detection_norm[0] * orig_w
                detection_y = detection_norm[1] * orig_h
                detection = (detection_x, detection_y)

                # 默认置信度
                confidence = 0.8 if frame_count > SEQ_LEN else 0.5

            # 使用卡尔曼滤波进行跟踪融合
            if use_kalman and kalman_tracker is not None:
                tracked_pos, status = kalman_tracker.update(detection, confidence)

                if tracked_pos is not None:
                    final_x, final_y = tracked_pos
                    stats['tracked_frames'] += 1

                    # 根据跟踪状态调整显示颜色
                    if status == 'tracking':
                        color = (0, 255, 0)  # 绿色 - 正常跟踪
                    elif status == 'recovered':
                        color = (255, 255, 0)  # 青色 - 重新捕获
                    else:
                        color = (0, 0, 255)  # 红色 - 丢失
                else:
                    final_x, final_y = detection_x, detection_y
                    color = (0, 0, 255)
                    stats['lost_frames'] += 1
            else:
                final_x, final_y = detection_x, detection_y
                color = (0, 255, 0)
                stats['processed_frames'] += 1

            # 绘制结果
            final_x, final_y = int(final_x), int(final_y)

            # 绘制瞳孔点
            cv2.circle(frame, (final_x, final_y), 8, color, -1)
            cv2.circle(frame, (final_x, final_y), 15, color, 2)
            cv2.circle(frame, (final_x, final_y), 25, (255, 255, 255), 1)

            # 显示坐标和状态
            info_text = f"Pupil: ({final_x}, {final_y})"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if use_kalman and kalman_tracker is not None:
                state = kalman_tracker.get_state()
                if state:
                    status_text = f"Status: {'Tracking' if state['tracking'] else 'Lost'}"
                    vel = np.sqrt(state['velocity'][0] ** 2 + state['velocity'][1] ** 2)
                    cv2.putText(frame, f"Vel: {vel:.1f}", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, status_text, (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 添加进度条
            progress = (frame_count / total_frames) * 100
            progress_bar_y = orig_h - 10
            progress_bar_width = int(orig_w * 0.3)
            cv2.rectangle(frame, (10, progress_bar_y - 20),
                          (10 + progress_bar_width, progress_bar_y), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, progress_bar_y - 20),
                          (10 + int(progress_bar_width * progress / 100), progress_bar_y), (0, 255, 0), -1)
            cv2.putText(frame, f"{progress:.1f}%", (10 + progress_bar_width + 15, progress_bar_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 移除第一帧，保持滑动窗口
            frame_buffer.pop(0)

            out.write(frame)
            frame_count += 1
            stats['total_frames'] += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

    cap.release()
    out.release()

    # 计算统计信息
    if stats['total_frames'] > 0:
        stats['success_rate'] = stats['tracked_frames'] / stats['total_frames'] * 100
    else:
        stats['success_rate'] = 0

    print(f"\nProcessing completed!")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Tracked frames: {stats['tracked_frames']}")
    print(f"Success rate: {stats['success_rate']:.2f}%")

    return stats


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """上传视频并处理"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_filename = f"{timestamp}_{unique_id}_{filename}"

        # 保存上传文件
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)

        # 生成输出文件名
        output_filename = f"processed_{input_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        try:
            # 处理视频
            use_kalman = request.form.get('use_kalman', 'true').lower() == 'true'
            stats = process_video(input_path, output_path, use_kalman)

            # 返回结果
            return jsonify({
                'success': True,
                'message': 'Video processed successfully',
                'download_url': f'/download/{output_filename}',
                'stats': stats,
                'filename': output_filename
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/download/<filename>')
def download_video(filename):
    """下载处理后的视频"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route('/cleanup')
def cleanup():
    """清理旧文件"""
    import time

    max_age = 3600  # 1 小时后清理

    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if time.time() - os.path.getctime(file_path) > max_age:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    return jsonify({'message': 'Cleanup completed'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
