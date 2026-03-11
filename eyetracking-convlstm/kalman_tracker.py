import numpy as np
import cv2


class KalmanFilter:
    """卡尔曼滤波器用于瞳孔跟踪平滑"""

    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        """
        初始化卡尔曼滤波器

        状态向量：[x, y, vx, vy] (位置 + 速度)
        观测向量：[x, y] (检测到的瞳孔位置)
        """
        self.kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵 (恒定速度模型)
        dt = 1.0  # 时间间隔
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 观测矩阵
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 过程噪声协方差
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 观测噪声协方差
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # 初始状态协方差
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False
        self.last_measurement = None

    def predict(self):
        """预测下一帧位置"""
        if not self.initialized:
            return None

        prediction = self.kf.predict()
        return prediction.reshape(2)

    def update(self, measurement):
        """用观测值更新状态"""
        if measurement is None:
            return self.predict()

        measurement = np.array(measurement, dtype=np.float32).reshape(2, 1)

        if not self.initialized:
            # 初始化：直接使用第一个观测值
            self.kf.statePre[:2] = measurement
            self.kf.statePost[:2] = measurement
            self.kf.errorCovPre[:2, :2] = np.eye(2, dtype=np.float32) * 1.0
            self.kf.errorCovPost[:2, :2] = np.eye(2, dtype=np.float32) * 1.0
            self.initialized = True
            return measurement.reshape(2)

        # 先预测
        self.kf.predict()

        # 再更新
        self.kf.correct(measurement)

        return self.kf.statePost[:2].reshape(2)

    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.last_measurement = None


class AdaptiveKalmanTracker:
    """自适应卡尔曼跟踪器"""

    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.kf = KalmanFilter(process_noise, measurement_noise)
        self.consecutive_misses = 0
        self.max_consecutive_misses = 10  # 最大连续丢失帧数
        self.tracking = False

        # 自适应参数
        self.base_process_noise = process_noise
        self.base_measurement_noise = measurement_noise
        self.velocity_threshold = 5.0  # 速度阈值，用于调整噪声

    def update(self, detection, confidence=None):
        """
        更新跟踪器

        Args:
            detection: 检测到的瞳孔位置 (x, y)
            confidence: 检测置信度 (0-1)，None 表示未检测到

        Returns:
            tracked_position: 跟踪结果
            status: 跟踪状态 ('tracking', 'lost', 'recovered')
        """
        if detection is not None and (confidence is None or confidence > 0.5):
            # 有可靠检测
            if not self.tracking:
                # 重新初始化
                self.kf.reset()
                self.kf.update(detection)
                self.tracking = True
                self.consecutive_misses = 0
                return detection, 'recovered'
            else:
                # 正常更新
                self.consecutive_misses = 0

                # 自适应调整噪声
                velocity = self._estimate_velocity(detection)
                if velocity > self.velocity_threshold:
                    # 快速运动时增加过程噪声
                    self.kf.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.base_process_noise * 2.0
                else:
                    self.kf.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.base_process_noise

                tracked = self.kf.update(detection)
                return tracked, 'tracking'

        else:
            # 无检测或置信度低
            self.consecutive_misses += 1

            if self.consecutive_misses > self.max_consecutive_misses:
                self.tracking = False
                return None, 'lost'

            # 使用预测值
            predicted = self.kf.predict()
            if predicted is not None:
                return predicted, 'tracking'
            else:
                return None, 'lost'

    def _estimate_velocity(self, detection):
        """估计当前速度"""
        if not self.kf.initialized:
            return 0.0

        state = self.kf.kf.statePost
        vx = state[2, 0]
        vy = state[3, 0]
        return np.sqrt(vx ** 2 + vy ** 2)

    def get_state(self):
        """获取当前状态"""
        if not self.kf.initialized:
            return None

        state = self.kf.kf.statePost
        return {
            'position': state[:2].reshape(2),
            'velocity': state[2:].reshape(2),
            'tracking': self.tracking,
            'consecutive_misses': self.consecutive_misses
        }
