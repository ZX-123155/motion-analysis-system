"""
数据预处理与特征工程模块
- 滤波处理（巴特沃斯低通/高通滤波器）
- 时域特征提取
- 频域特征提取
- 滑窗分割
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from pathlib import Path


class SensorDataProcessor:
    """IMU传感器数据处理器"""

    SAMPLE_RATE = 50  # Hz
    WINDOW_SIZE = 128  # 约2.56秒窗口（50Hz * 2.56s）
    STEP_SIZE = 64     # 50% 重叠

    def __init__(self, low_cut=0.5, high_cut=20.0):
        self.low_cut = low_cut
        self.high_cut = high_cut

    def butter_lowpass(self, cutoff, order=4):
        """巴特沃斯低通滤波器"""
        nyquist = 0.5 * self.SAMPLE_RATE
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def butter_highpass(self, cutoff, order=4):
        """巴特沃斯高通滤波器"""
        nyquist = 0.5 * self.SAMPLE_RATE
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def butter_bandpass(self, lowcut, highcut, order=4):
        """巴特沃斯带通滤波器"""
        nyquist = 0.5 * self.SAMPLE_RATE
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def apply_filter(self, signal, filter_type="bandpass"):
        """对信号应用滤波器"""
        if filter_type == "lowpass":
            b, a = self.butter_lowpass(self.high_cut)
        elif filter_type == "highpass":
            b, a = self.butter_highpass(self.low_cut)
        else:  # bandpass
            b, a = self.butter_bandpass(self.low_cut, self.high_cut)
        return filtfilt(b, a, signal)

    def filter_dataframe(self, df, columns=None):
        """对DataFrame中的所有传感器列进行滤波"""
        if columns is None:
            columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

        df_filtered = df.copy()
        for col in columns:
            if col in df.columns:
                df_filtered[col] = self.apply_filter(df[col].values)
        return df_filtered

    def extract_temporal_features(self, signal):
        """提取时域特征"""
        features = {
            "mean": np.mean(signal),
            "std": np.std(signal),
            "max": np.max(signal),
            "min": np.min(signal),
            "range": np.max(signal) - np.min(signal),
            "rms": np.sqrt(np.mean(signal**2)),
            "skewness": skew(signal),
            "kurtosis": kurtosis(signal),
            "energy": np.sum(signal**2),
            "peak_to_peak": np.ptp(signal),
        }

        # 过零率
        zero_crossings = np.sum(np.diff(np.signbit(signal - np.mean(signal))))
        features["zero_crossing_rate"] = zero_crossings / len(signal)

        # 信号幅值面积 (SMA)
        features["sma"] = np.sum(np.abs(signal)) / len(signal)

        return features

    def extract_frequency_features(self, signal):
        """提取频域特征"""
        n = len(signal)
        fft_vals = np.abs(fft(signal))[:n // 2]
        freqs = fftfreq(n, 1 / self.SAMPLE_RATE)[:n // 2]

        total_power = np.sum(fft_vals**2)
        if total_power == 0:
            total_power = 1e-10

        # 频谱质心
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)

        # 主频
        dominant_freq = freqs[np.argmax(fft_vals)]

        # 频带能量分布
        bands = {
            "0-2Hz": (0, 2),
            "2-5Hz": (2, 5),
            "5-10Hz": (5, 10),
            "10-20Hz": (10, 20),
        }
        band_energy = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy[f"energy_{band_name}"] = np.sum(fft_vals[mask]**2) / total_power

        features = {
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "spectral_energy": total_power,
            **band_energy,
        }
        return features

    def extract_window_features(self, window_df):
        """从单个窗口提取所有特征"""
        all_features = {}
        sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                       "acc_magnitude", "gyro_magnitude"]

        for col in sensor_cols:
            if col not in window_df.columns:
                continue
            signal = window_df[col].values
            temporal = self.extract_temporal_features(signal)
            freq = self.extract_frequency_features(signal)

            for k, v in temporal.items():
                all_features[f"{col}_{k}"] = v
            for k, v in freq.items():
                all_features[f"{col}_{k}"] = v

        # 标签
        if "label" in window_df.columns:
            all_features["label"] = window_df["label"].iloc[0]

        return all_features

    def sliding_window(self, df, window_size=None, step_size=None):
        """滑窗分割，返回特征列表"""
        if window_size is None:
            window_size = self.WINDOW_SIZE
        if step_size is None:
            step_size = self.STEP_SIZE

        features_list = []
        n = len(df)

        for start in range(0, n - window_size + 1, step_size):
            window = df.iloc[start:start + window_size]
            feat = self.extract_window_features(window)
            feat["window_start"] = start
            feat["window_end"] = start + window_size
            feat["sample_id"] = window["sample_id"].iloc[0] if "sample_id" in df.columns else None
            features_list.append(feat)

        return pd.DataFrame(features_list)

    def process_dataset(self, imu_df):
        """处理整个数据集：滤波 -> 滑窗 -> 特征提取"""
        # 1. 滤波
        print("  Applying filters...")
        df_filtered = self.filter_dataframe(imu_df)

        # 2. 按sample_id分组处理
        print("  Sliding window feature extraction...")
        all_features = []
        for sample_id, group in df_filtered.groupby("sample_id"):
            features_df = self.sliding_window(group)
            all_features.append(features_df)

        features_df = pd.concat(all_features, ignore_index=True)
        print(f"  Extracted {len(features_df)} windows with "
              f"{len(features_df.columns) - 4} features each")
        return features_df


def main():
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    imu_df = pd.read_csv(raw_dir / "imu_data.csv")

    processor = SensorDataProcessor(low_cut=0.5, high_cut=20.0)
    features_df = processor.process_dataset(imu_df)

    features_df.to_csv(processed_dir / "features.csv", index=False)
    print(f"\nFeatures saved to {processed_dir / 'features.csv'}")
    print(f"Feature dimensions: {features_df.shape}")
    print(f"Labels distribution:\n{features_df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
