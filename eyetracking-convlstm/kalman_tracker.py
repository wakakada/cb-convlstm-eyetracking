# 自适应卡尔曼滤波器
import numpy as np
import cv2


class KalmanFilter:
    """卡尔曼滤波器用于瞳孔跟踪平滑"""
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        """
        初始化卡尔曼滤波器
        parameters:
            process_noise:过程噪声协方差的缩放系数，反映模型的不可靠程度
            measurement_noise:测量噪声协方差的缩放系数，反映检测器输出位置的精度
        状态向量：[x, y, v_x, v_y] (位置 + 速度)
        观测向量：[x, y] (检测到的瞳孔位置)
        """
        self.kf = cv2.KalmanFilter(4, 2)    # 状态空间为4维，观测空间为2维

        # 状态转移矩阵 (恒定速度模型)
        dt = 1.0  # 视频帧之间的固定时间间隔
        self.kf.transitionMatrix = np.array([   # 描述状态如何从上一时刻演化到当前时刻
            [1, 0, dt, 0],  # 新x=旧x+dt*v_x
            [0, 1, 0, dt],  # 新y=旧y+dt*v_y
            [0, 0, 1, 0],   # 新v_x=旧v_x
            [0, 0, 0, 1]    # 新v_y=旧v_y
        ], dtype=np.float32)

        # 观测矩阵
        self.kf.measurementMatrix = np.array([  # 将状态空间映射到观测空间
            [1, 0, 0, 0],   # 提取x位置
            [0, 1, 0, 0]    # 提取y位置
        ], dtype=np.float32)

        # 过程噪声协方差，值越大，表示模型预测越不可靠，表示模型本身的不确定性或误差
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 观测噪声协方差，值越大，表示测量结果越不准确，表示传感器或检测器的不确定性
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # 初始状态协方差，表示初始状态估计的不确定性，值为1.0表示初始时我们对状态估计不太确定。随着滤波器运行，这个矩阵会不断更新，反映当前估计的准确性。
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False    # 跟踪器是否已初始化

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

        measurement = np.array(measurement, dtype=np.float32).reshape(2, 1) # 转换成列向量以满足OpenCV接口要求

        if not self.initialized:
            # 初始化：直接使用第一个观测值
            self.kf.statePre[:2] = measurement
            self.kf.statePost[:2] = measurement
            # 设置初始状态协方差为1.0，表示对初试位置有一定把握但非完全确定
            self.kf.errorCovPre[:2, :2] = np.eye(2, dtype=np.float32) * 1.0
            self.kf.errorCovPost[:2, :2] = np.eye(2, dtype=np.float32) * 1.0
            self.initialized = True
            return measurement.reshape(2)

        # 先预测
        self.kf.predict()

        # 再用预测值进行更新
        self.kf.correct(measurement)

        return self.kf.statePost[:2].reshape(2)

    def reset(self):
        """重置滤波器，使滤波器可重新开始跟踪（例如跟踪目标丢失后重新捕获）"""
        self.initialized = False


class AdaptiveKalmanTracker:
    """自适应卡尔曼跟踪器，在基础卡尔曼滤波器上增加了自适应机制与丢失处理逻辑"""
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.kf = KalmanFilter(process_noise, measurement_noise)    # 接收过程噪声和测量噪声的基础值
        self.consecutive_misses = 0
        self.max_consecutive_misses = 10  # 最大连续丢失帧数，超过此阈值认为目标彻底丢失
        self.tracking = False

        # 保存基础噪声参数，用于后续动态调整
        self.base_process_noise = process_noise
        self.base_measurement_noise = measurement_noise
        self.velocity_threshold = 5.0  # 速度阈值，估计速度超过该值时，增加过程噪声以应对快速运动

    def update(self, detection, confidence=None):
        """
        更新跟踪器

        Args:
            detection: 检测到的瞳孔位置 (x, y)
            confidence: 检测置信度 (0-1)，用于判断检测可靠性，None 表示未检测到

        Returns:
            tracked_position: 跟踪结果
            status: 跟踪状态 ('tracking', 'lost', 'recovered')
        """
        if detection is not None and (confidence is None or confidence > 0.5):
            # 有可靠检测
            if not self.tracking:
                # 当前未处于跟踪状态，重新初始化滤波器
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
                    # 快速运动时增加过程噪声使滤波器更信任观测值
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

        state = self.kf.kf.statePost    # 从滤波器的后验状态向量中提取速度分量（vx, vy）并计算合速度大小
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
