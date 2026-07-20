"""
运动数据模拟生成器
生成四种运动类型（走路、跑步、跳跃、高抬腿）的IMU传感器数据和GNSS轨迹数据
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


class MotionDataGenerator:
    """模拟人体运动的IMU和GNSS数据"""

    SAMPLE_RATE = 50  # Hz
    ACTIVITIES = ["walking", "running", "jumping", "high_knees"]

    def __init__(self, duration_sec=10, seed=42):
        self.duration = duration_sec
        self.n_samples = int(duration_sec * self.SAMPLE_RATE)
        self.rng = np.random.RandomState(seed)
        self.t = np.linspace(0, duration_sec, self.n_samples, endpoint=False)

    def _add_noise(self, signal, snr_db=30):
        """给信号添加高斯噪声"""
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise

    def _base_oscillation(self, freq, amplitude, phase=0, duty_cycle=1.0):
        """生成基础振荡信号"""
        raw = amplitude * np.sin(2 * np.pi * freq * self.t + phase)
        # 模拟步态的非对称性
        envelope = 0.5 * (1 + np.sin(2 * np.pi * freq * self.t / 2))
        raw = raw * (1 + 0.3 * envelope)
        return raw

    def generate_walking(self):
        """走路：约1.5-2 Hz的步频，加速度幅值中等，周期性明显"""
        step_freq = 1.8  # Hz
        # 加速度 (m/s^2)
        acc_x = self._base_oscillation(step_freq, 1.5, phase=0)
        acc_y = self._base_oscillation(step_freq, 2.5, phase=np.pi / 2)
        acc_z = self._base_oscillation(step_freq, 1.2, phase=np.pi) + 9.8  # 重力

        # 角速度 (rad/s)
        gyro_x = self._base_oscillation(step_freq, 0.4, phase=0.3)
        gyro_y = self._base_oscillation(step_freq, 0.3, phase=1.0)
        gyro_z = self._base_oscillation(step_freq, 0.2, phase=0.6)

        # 加噪声
        acc_x = self._add_noise(acc_x, snr_db=28)
        acc_y = self._add_noise(acc_y, snr_db=28)
        acc_z = self._add_noise(acc_z, snr_db=28)
        gyro_x = self._add_noise(gyro_x, snr_db=25)
        gyro_y = self._add_noise(gyro_y, snr_db=25)
        gyro_z = self._add_noise(gyro_z, snr_db=25)

        return self._build_df(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, "walking")

    def generate_running(self):
        """跑步：约2.5-3 Hz的步频，加速度幅值大，冲击特征明显"""
        step_freq = 2.8
        # 加速度 - 更大的幅值，更明显的冲击
        acc_x = self._base_oscillation(step_freq, 3.0, phase=0)
        acc_y = self._base_oscillation(step_freq, 5.0, phase=np.pi / 2)
        acc_z_raw = self._base_oscillation(step_freq, 3.5, phase=np.pi)

        # 跑步的冲击特征：触地时有明显的尖峰
        impact = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            phase_i = (step_freq * self.t[i]) % 1.0
            if phase_i < 0.1:
                impact[i] = 8.0 * (1 - phase_i / 0.1)

        acc_z = acc_z_raw + impact + 9.8

        gyro_x = self._base_oscillation(step_freq, 0.8, phase=0.3)
        gyro_y = self._base_oscillation(step_freq, 0.6, phase=1.0)
        gyro_z = self._base_oscillation(step_freq, 0.5, phase=0.6)

        acc_x = self._add_noise(acc_x, snr_db=30)
        acc_y = self._add_noise(acc_y, snr_db=30)
        acc_z = self._add_noise(acc_z, snr_db=30)
        gyro_x = self._add_noise(gyro_x, snr_db=25)
        gyro_y = self._add_noise(gyro_y, snr_db=25)
        gyro_z = self._add_noise(gyro_z, snr_db=25)

        return self._build_df(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, "running")

    def generate_jumping(self):
        """跳跃：约1-2 Hz，有明显的腾空和落地阶段，加速度变化剧烈"""
        jump_freq = 1.5

        # 跳跃模式：起跳+腾空+落地
        acc_y = np.zeros(self.n_samples)
        acc_z = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            phase_i = (jump_freq * self.t[i]) % 1.0
            if phase_i < 0.15:
                acc_y[i] = 3.0 + 2.0 * np.sin(phase_i / 0.15 * np.pi / 2)
                acc_z[i] = 15.0 + np.sin(phase_i / 0.15 * np.pi)
            elif phase_i < 0.3:
                acc_y[i] = 2.0
                acc_z[i] = 5.0  # 腾空（低重力感）
            elif phase_i < 0.5:
                acc_z[i] = -3.0 + 18.0 * (phase_i - 0.3) / 0.2
            else:
                acc_y[i] = 0.2 * np.sin(2 * np.pi * 3 * phase_i)
                acc_z[i] = 9.8 + 0.5 * np.sin(2 * np.pi * 3 * phase_i)

        acc_x = self._base_oscillation(jump_freq, 0.8, phase=0)

        gyro_x = np.zeros_like(self.t)
        gyro_y = np.zeros_like(self.t)
        gyro_z = np.zeros_like(self.t)
        for i in range(self.n_samples):
            phase_i = (jump_freq * self.t[i]) % 1.0
            if phase_i < 0.2:
                gyro_x[i] = 1.5 * np.sin(phase_i / 0.2 * np.pi)
                gyro_y[i] = 1.0 * np.sin(phase_i / 0.2 * np.pi)
            else:
                gyro_x[i] = 0.2 * np.sin(2 * np.pi * 4 * phase_i)
                gyro_y[i] = 0.15 * np.sin(2 * np.pi * 4 * phase_i)

        gyro_z = self._base_oscillation(jump_freq, 0.3, phase=0.5)

        acc_x = self._add_noise(acc_x, snr_db=28)
        acc_y = self._add_noise(acc_y, snr_db=28)
        acc_z = self._add_noise(acc_z, snr_db=28)
        gyro_x = self._add_noise(gyro_x, snr_db=25)
        gyro_y = self._add_noise(gyro_y, snr_db=25)
        gyro_z = self._add_noise(gyro_z, snr_db=25)

        return self._build_df(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, "jumping")

    def generate_high_knees(self):
        """高抬腿：约2-3 Hz，大腿大幅摆动，垂直方向周期性变化明显"""
        step_freq = 2.5
        # 高抬腿特征：大腿抬高时角速度大，垂直方向有明显节奏
        acc_x = self._base_oscillation(step_freq, 1.0, phase=0)
        acc_y = self._base_oscillation(step_freq, 3.5, phase=0.5)
        acc_z = self._base_oscillation(step_freq, 4.0, phase=np.pi) + 9.8

        # 高抬腿角速度特征：大幅度的交替摆动
        gyro_x = 1.5 * np.sin(2 * np.pi * step_freq * self.t)
        gyro_y = 1.2 * np.sin(2 * np.pi * step_freq * self.t + np.pi / 2)
        gyro_z = self._base_oscillation(step_freq * 2, 0.6, phase=0.3)

        acc_x = self._add_noise(acc_x, snr_db=28)
        acc_y = self._add_noise(acc_y, snr_db=28)
        acc_z = self._add_noise(acc_z, snr_db=28)
        gyro_x = self._add_noise(gyro_x, snr_db=25)
        gyro_y = self._add_noise(gyro_y, snr_db=25)
        gyro_z = self._add_noise(gyro_z, snr_db=25)

        return self._build_df(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, "high_knees")

    def _build_df(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label):
        """构建DataFrame"""
        df = pd.DataFrame({
            "timestamp": self.t,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,
            "gyro_z": gyro_z,
            "label": label,
        })
        # 添加合成量
        df["acc_magnitude"] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        df["gyro_magnitude"] = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        return df

    def generate_trajectory(self, activity="walking", start_lat=39.9042, start_lon=116.4074):
        """生成模拟GNSS轨迹数据"""
        n = self.n_samples
        dt = 1.0 / self.SAMPLE_RATE

        speeds = {
            "walking": 1.4,   # m/s (~5 km/h)
            "running": 3.5,   # m/s (~12.6 km/h)
            "jumping": 0.8,
            "high_knees": 1.0,
        }
        speed = speeds.get(activity, 1.0)
        heading = self.rng.uniform(0, 2 * np.pi)
        heading += self.rng.normal(0, 0.01, n).cumsum() * 0.1  # 平滑的方向变化

        # 经纬度转换（粗略）
        lat_step = speed * np.cos(heading) * dt / 111320.0
        lon_step = speed * np.sin(heading) * dt / (111320.0 * np.cos(np.radians(start_lat)))

        lat = start_lat + np.cumsum(lat_step)
        lon = start_lon + np.cumsum(lon_step)
        lat += self.rng.normal(0, 2e-6, n)  # GPS噪声
        lon += self.rng.normal(0, 2e-6, n)

        trajectory = pd.DataFrame({
            "timestamp": self.t,
            "latitude": lat,
            "longitude": lon,
            "speed": speed + self.rng.normal(0, 0.1, n),
            "heading": np.degrees(heading),
        })
        return trajectory

    def generate_full_dataset(self, samples_per_activity=20):
        """生成完整数据集"""
        all_imu = []
        all_traj = []
        metadata = []

        for act in self.ACTIVITIES:
            for i in range(samples_per_activity):
                self.__init__(duration_sec=self.rng.uniform(8, 15), seed=42 + i)
                gen_func = getattr(self, f"generate_{act}")
                df_imu = gen_func()
                df_traj = self.generate_trajectory(act)

                # 重置duration相关属性以便下一个样本
                sample_id = f"{act}_{i:03d}"
                df_imu["sample_id"] = sample_id
                df_traj["sample_id"] = sample_id
                all_imu.append(df_imu)
                all_traj.append(df_traj)
                metadata.append({
                    "sample_id": sample_id,
                    "activity": act,
                    "duration_sec": self.duration,
                    "sample_rate_hz": self.SAMPLE_RATE,
                })

        imu_data = pd.concat(all_imu, ignore_index=True)
        traj_data = pd.concat(all_traj, ignore_index=True)
        meta_df = pd.DataFrame(metadata)

        return imu_data, traj_data, meta_df


def main():
    output_dir = Path(__file__).parent.parent / "data"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    generator = MotionDataGenerator(duration_sec=10, seed=42)
    imu_data, traj_data, meta_df = generator.generate_full_dataset(samples_per_activity=15)

    imu_data.to_csv(raw_dir / "imu_data.csv", index=False)
    traj_data.to_csv(raw_dir / "trajectory_data.csv", index=False)
    meta_df.to_csv(raw_dir / "metadata.csv", index=False)

    print(f"Generated {len(meta_df)} motion samples:")
    print(meta_df["activity"].value_counts().to_string())
    print(f"IMU data shape: {imu_data.shape}")
    print(f"Trajectory data shape: {traj_data.shape}")
    print(f"Data saved to {raw_dir}")


if __name__ == "__main__":
    main()
