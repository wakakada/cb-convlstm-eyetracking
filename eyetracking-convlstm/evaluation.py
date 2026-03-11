"""
评估卡尔曼滤波对跟踪性能的提升
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kalman_tracker import AdaptiveKalmanTracker


def evaluate_kalman_effect(detection_sequence, ground_truth=None):
    """
    评估卡尔曼滤波的效果

    Args:
        detection_sequence: 检测序列 [(x1, y1), (x2, y2), ...]
        ground_truth: 真实值序列 (可选)
    """
    kalman = AdaptiveKalmanTracker()

    original_x, original_y = [], []
    tracked_x, tracked_y = [], []

    for det in detection_sequence:
        original_x.append(det[0])
        original_y.append(det[1])

        tracked, _ = kalman.update(det)
        if tracked is not None:
            tracked_x.append(tracked[0])
            tracked_y.append(tracked[1])
        else:
            tracked_x.append(original_x[-1])
            tracked_y.append(original_y[-1])

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(original_x, label='Original Detection', alpha=0.5)
    plt.plot(tracked_x, label='Kalman Tracked', linewidth=2)
    if ground_truth:
        plt.plot([gt[0] for gt in ground_truth], label='Ground Truth', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('X Coordinate')
    plt.legend()
    plt.title('X Coordinate Comparison')

    plt.subplot(1, 2, 2)
    plt.plot(original_y, label='Original Detection', alpha=0.5)
    plt.plot(tracked_y, label='Kalman Tracked', linewidth=2)
    if ground_truth:
        plt.plot([gt[1] for gt in ground_truth], label='Ground Truth', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Y Coordinate Comparison')

    plt.tight_layout()
    plt.savefig('kalman_comparison.png')
    print("Comparison plot saved to kalman_comparison.png")


if __name__ == "__main__":
    # 示例：生成带噪声的检测序列
    np.random.seed(42)
    frames = 100

    # 模拟真实的瞳孔运动轨迹
    true_x = np.sin(np.linspace(0, 2 * np.pi, frames)) * 50 + 100
    true_y = np.cos(np.linspace(0, 2 * np.pi, frames)) * 50 + 100

    # 添加噪声模拟检测误差
    noisy_x = true_x + np.random.randn(frames) * 5
    noisy_y = true_y + np.random.randn(frames) * 5

    detections = list(zip(noisy_x, noisy_y))
    ground_truth = list(zip(true_x, true_y))

    evaluate_kalman_effect(detections, ground_truth)
