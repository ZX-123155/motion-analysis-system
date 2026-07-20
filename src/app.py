"""
运动数据分析系统 - Flask Web应用
- REST API：静态分析接口
- WebSocket：实时数据流处理与展示
- 运动识别推理
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import time
import threading
from scipy.fft import fft, fftfreq
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

from data_generator import MotionDataGenerator
from preprocessing import SensorDataProcessor


_PROJECT_ROOT = Path(__file__).parent.parent
app = Flask(__name__,
            template_folder=str(_PROJECT_ROOT / "templates"),
            static_folder=str(_PROJECT_ROOT / "static"))
app.config["SECRET_KEY"] = "motion-analysis-secret"


@app.after_request
def disable_cache(response):
    """禁用浏览器缓存，确保每次刷新获取最新页面"""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# 全局变量：加载预训练模型和处理器
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

classifier = None
scaler = None
label_encoder = None
feature_names = None
processor = None

# 实时数据状态
stream_state = {
    "running": False,
    "current_activity": "unknown",
    "lat": 39.9042,
    "lon": 116.4074,
    "step_freq": 0,
    "speed": 0,
    "confidence": 0,
    "buffer": [],
    "history": [],
}

# 手机端真实数据状态
mobile_state = {
    "connected": False,
    "buffer": [],
    "buffer_size": 128,     # 128个采样点做一次预测
    "overlap": 64,          # 50%重叠
    "lat": None,
    "lon": None,
    "sample_count": 0,
    "last_prediction_time": 0,
    "prediction_cooldown": 1.0,  # 每秒最多预测一次
}


def load_model():
    """加载预训练模型"""
    global classifier, scaler, label_encoder, feature_names, processor

    try:
        classifier = joblib.load(MODEL_DIR / "classifier.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        processor = SensorDataProcessor(low_cut=0.5, high_cut=20.0)
        print("Model loaded successfully")
        return True
    except FileNotFoundError:
        print("Model not found. Training may be required.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def predict_activity(sensor_data_df):
    """对传感器数据窗口进行活动预测"""
    if classifier is None or scaler is None:
        return "unknown", 0.0

    # 提取特征
    features = processor.extract_window_features(sensor_data_df)
    X = np.array([[features[name] for name in feature_names]])

    # 处理NaN
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    try:
        X_scaled = scaler.transform(X)
        probs = classifier.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(probs)
        activity = label_encoder.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        return activity, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown", 0.0


# ====== REST API ======

@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/mobile")
def mobile():
    """手机端传感器采集页面"""
    return render_template("mobile.html")


@app.route("/api/status")
def api_status():
    """系统状态"""
    model_loaded = classifier is not None
    return jsonify({
        "model_loaded": model_loaded,
        "activities": ["walking", "running"],
        "stream_running": stream_state["running"],
    })


@app.route("/api/dataset/info")
def api_dataset_info():
    """数据集信息"""
    try:
        meta = pd.read_csv(DATA_DIR / "raw" / "metadata.csv")
        imu = pd.read_csv(DATA_DIR / "raw" / "imu_data.csv")

        info = {
            "total_samples": len(meta),
            "total_imu_points": len(imu),
            "activities": meta["activity"].value_counts().to_dict(),
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/sample/<sample_id>")
def api_sample_data(sample_id):
    """获取单个样本数据"""
    try:
        imu = pd.read_csv(DATA_DIR / "raw" / "imu_data.csv")
        traj = pd.read_csv(DATA_DIR / "raw" / "trajectory_data.csv")

        imu_sample = imu[imu["sample_id"] == sample_id]
        traj_sample = traj[traj["sample_id"] == sample_id]

        return jsonify({
            "sample_id": sample_id,
            "imu": imu_sample.to_dict(orient="records")[:500],
            "trajectory": traj_sample.to_dict(orient="records")[:500],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/metrics")
def api_model_metrics():
    """模型评估指标"""
    try:
        with open(MODEL_DIR / "metrics.json", "r") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/static/figures/<filename>")
def serve_figure(filename):
    """提供图表图片"""
    return send_from_directory(Path(__file__).parent.parent / "static" / "figures", filename)


# ====== WebSocket ======

@socketio.on("connect")
def handle_connect():
    print(f"Client connected")
    emit("status", {"message": "Connected to motion analysis server"})


@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client disconnected")
    stream_state["running"] = False


@socketio.on("start_stream")
def handle_start_stream(data=None):
    """启动实时数据流模拟"""
    if stream_state["running"]:
        emit("warning", {"message": "Stream already running"})
        return

    stream_state["running"] = True
    stream_state["buffer"] = []
    stream_state["history"] = []

    activity_sequence = data.get("activity_sequence", ["walking", "running"]) if data else ["walking", "running"]
    emit("status", {"message": f"Stream started with sequence: {activity_sequence}"})

    # 后台线程生成数据
    def stream_thread():
        gen = MotionDataGenerator(duration_sec=4, seed=None)
        processor_local = SensorDataProcessor(low_cut=0.5, high_cut=20.0)
        step = 0

        while stream_state["running"]:
            try:
                # 循环运动类型
                act = activity_sequence[(step // 4) % len(activity_sequence)]

                gen_func = getattr(gen, f"generate_{act}")
                df = gen_func()

                # 滤波
                df_filt = processor_local.filter_dataframe(df)

                # 生成轨迹点
                traj = gen.generate_trajectory(act,
                    start_lat=stream_state["lat"],
                    start_lon=stream_state["lon"])

                last_traj = traj.iloc[-1]
                stream_state["lat"] = float(last_traj["latitude"])
                stream_state["lon"] = float(last_traj["longitude"])
                stream_state["speed"] = float(last_traj["speed"])

                # 累积缓冲
                stream_state["buffer"].extend(df_filt.to_dict(orient="records"))

                # 有足够数据时进行预测
                buffer_len = len(stream_state["buffer"])
                if buffer_len >= 128:
                    window_df = pd.DataFrame(stream_state["buffer"][-128:])
                    pred_act, conf = predict_activity(window_df)
                    stream_state["current_activity"] = pred_act
                    stream_state["confidence"] = conf

                    # 步频估计
                    if "acc_magnitude" in window_df.columns:
                        acc_mag = window_df["acc_magnitude"].values
                        n = len(acc_mag)
                        fft_vals = np.abs(fft(acc_mag))[:n // 2]
                        freqs = fftfreq(n, 1 / 50)[:n // 2]
                        mask = (freqs >= 0.5) & (freqs <= 5.0)
                        if np.any(mask):
                            stream_state["step_freq"] = float(freqs[mask][np.argmax(fft_vals[mask])])
                        else:
                            stream_state["step_freq"] = 0

                    # 仅保留最近的数据
                    stream_state["buffer"] = stream_state["buffer"][-256:]

                # 缩采样发送（每10个点发一个，降低带宽）
                send_data = df_filt.iloc[::5].copy()
                send_data["timestamp"] = send_data["timestamp"].tolist()
                sensor_payload = {
                    "acc_x": send_data["acc_x"].tolist(),
                    "acc_y": send_data["acc_y"].tolist(),
                    "acc_z": send_data["acc_z"].tolist(),
                    "gyro_x": send_data["gyro_x"].tolist(),
                    "gyro_y": send_data["gyro_y"].tolist(),
                    "gyro_z": send_data["gyro_z"].tolist(),
                    "timestamps": send_data["timestamp"].tolist(),
                }

                socketio.emit("sensor_data", sensor_payload)

                # 发送轨迹更新
                socketio.emit("trajectory_update", {
                    "lat": stream_state["lat"],
                    "lon": stream_state["lon"],
                    "speed": stream_state["speed"],
                })

                # 发送识别结果
                socketio.emit("activity_update", {
                    "activity": stream_state["current_activity"],
                    "confidence": stream_state["confidence"],
                    "step_freq": stream_state["step_freq"],
                    "speed": stream_state["speed"],
                })

                step += 1
                time.sleep(3.5)  # 每3.5秒发送一个样本

            except Exception as e:
                print(f"Stream error: {e}")
                stream_state["running"] = False
                socketio.emit("error", {"message": str(e)})
                break

        stream_state["running"] = False
        socketio.emit("status", {"message": "Stream stopped"})

    thread = threading.Thread(target=stream_thread, daemon=True)
    thread.start()


@socketio.on("stop_stream")
def handle_stop_stream():
    """停止实时数据流"""
    stream_state["running"] = False
    emit("status", {"message": "Stream stopping..."})


@socketio.on("predict_single")
def handle_predict_single(data):
    """单次预测请求"""
    try:
        df = pd.DataFrame(data["sensor_data"])
        activity, confidence = predict_activity(df)
        emit("prediction_result", {
            "activity": activity,
            "confidence": confidence,
        })
    except Exception as e:
        emit("error", {"message": str(e)})


# ====== 手机端真实传感器数据接收 ======

@socketio.on("mobile_connect")
def handle_mobile_connect():
    """手机端连接"""
    mobile_state["connected"] = True
    mobile_state["buffer"] = []
    mobile_state["sample_count"] = 0
    mobile_state["lat"] = None
    mobile_state["lon"] = None
    print("[Mobile] Phone connected - real sensor streaming active")
    # 通知所有仪表盘客户端：真实数据源已连接
    socketio.emit("status", {"message": "手机传感器已连接，接收真实数据中..."})
    socketio.emit("data_source", {"source": "mobile"})
    emit("mobile_status", {"status": "connected"})


@socketio.on("mobile_disconnect")
def handle_mobile_disconnect():
    """手机端断开"""
    mobile_state["connected"] = False
    mobile_state["buffer"] = []
    print("[Mobile] Phone disconnected")
    socketio.emit("status", {"message": "手机传感器已断开"})
    socketio.emit("data_source", {"source": "simulated"})


@socketio.on("mobile_sensor_data")
def handle_mobile_sensor_data(data):
    """接收手机端原始传感器数据包"""
    if not mobile_state["connected"]:
        return

    # 数据包格式: { acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, lat, lon, speed, heading }
    try:
        acc_x = float(data.get("acc_x", 0))
        acc_y = float(data.get("acc_y", 0))
        acc_z = float(data.get("acc_z", 0))
        gyro_x = float(data.get("gyro_x", 0))
        gyro_y = float(data.get("gyro_y", 0))
        gyro_z = float(data.get("gyro_z", 0))
        ts = data.get("timestamp", time.time())
    except (ValueError, TypeError) as e:
        print(f"[Mobile] Invalid sensor data: {e}")
        return

    # 更新GPS
    if data.get("lat") is not None and data.get("lon") is not None:
        mobile_state["lat"] = float(data["lat"])
        mobile_state["lon"] = float(data["lon"])

    # 计算合成量
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    # 加入到缓冲区
    sample = {
        "timestamp": ts,
        "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
        "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z,
        "acc_magnitude": acc_mag, "gyro_magnitude": gyro_mag,
    }
    mobile_state["buffer"].append(sample)
    mobile_state["sample_count"] += 1

    # 实时转发传感器数据给仪表盘（缩采样，每2个点发1个）
    if mobile_state["sample_count"] % 2 == 0:
        socketio.emit("sensor_data", {
            "acc_x": [acc_x], "acc_y": [acc_y], "acc_z": [acc_z],
            "gyro_x": [gyro_x], "gyro_y": [gyro_y], "gyro_z": [gyro_z],
            "timestamps": [ts],
        })

    # 发送GPS更新
    if mobile_state["lat"] is not None:
        speed = float(data.get("speed", 0)) if data.get("speed") is not None else 0
        socketio.emit("trajectory_update", {
            "lat": mobile_state["lat"],
            "lon": mobile_state["lon"],
            "speed": speed,
        })

    # 缓冲区累积足够时进行预测
    buffer_len = len(mobile_state["buffer"])
    if buffer_len >= mobile_state["buffer_size"]:
        # 检查冷却时间
        now = time.time()
        if now - mobile_state["last_prediction_time"] < mobile_state["prediction_cooldown"]:
            return

        mobile_state["last_prediction_time"] = now

        # 取最近128个点
        window_data = mobile_state["buffer"][-mobile_state["buffer_size"]:]
        window_df = pd.DataFrame(window_data)

        # 预测
        activity, confidence = predict_activity(window_df)

        # 步频估计
        step_freq = 0
        if len(window_data) >= 64:
            acc_mag_vals = np.array([s["acc_magnitude"] for s in window_data])
            n = len(acc_mag_vals)
            fft_vals = np.abs(fft(acc_mag_vals))[:n // 2]
            freqs = fftfreq(n, 1 / 50)[:n // 2]
            mask = (freqs >= 0.5) & (freqs <= 5.0)
            if np.any(mask):
                step_freq = float(freqs[mask][np.argmax(fft_vals[mask])])

        # 广播给所有仪表盘客户端
        socketio.emit("activity_update", {
            "activity": activity,
            "confidence": confidence,
            "step_freq": step_freq,
            "speed": float(data.get("speed", 0)) if data.get("speed") else 0,
        })

        # 打印调试
        print(f"[Mobile] Predicted: {activity} (conf: {confidence:.2f}) | step_freq: {step_freq:.2f} Hz | buffer: {buffer_len}")

        # 缩小缓冲保留50%重叠
        mobile_state["buffer"] = mobile_state["buffer"][-mobile_state["buffer_size"] // 2:]


@socketio.on("mobile_gps_update")
def handle_mobile_gps(data):
    """手机端GPS位置更新"""
    if data.get("lat") and data.get("lon"):
        mobile_state["lat"] = float(data["lat"])
        mobile_state["lon"] = float(data["lon"])
        socketio.emit("trajectory_update", {
            "lat": mobile_state["lat"],
            "lon": mobile_state["lon"],
            "speed": float(data.get("speed", 0)),
        })


if __name__ == "__main__":
    model_loaded = load_model()
    if not model_loaded:
        print("WARNING: Model not loaded. Run train_model.py first.")

    print("\n" + "=" * 50)
    print("  Motion Analysis System")
    print("  仪表盘: http://localhost:5000")
    print("  手机端: http://<电脑IP>:5000/mobile")
    print("=" * 50 + "\n")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
