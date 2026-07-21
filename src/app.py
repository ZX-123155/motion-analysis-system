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


def server_heartbeat_thread():
    """后台线程：每2秒发送心跳，用于客户端快速检测服务器断开"""
    while True:
        time.sleep(2)
        try:
            socketio.emit("server_heartbeat")
        except Exception:
            break

heartbeat_thread = threading.Thread(target=server_heartbeat_thread, daemon=True)
heartbeat_thread.start()

# 全局变量：加载预训练模型和处理器
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

classifier = None
scaler = None
label_encoder = None
feature_names = None
processor = None

# 默认位置：上海杨浦区湾谷科技园A8
DEFAULT_LAT = 31.3382
DEFAULT_LON = 121.5098

# 实时数据状态
stream_state = {
    "running": False,
    "current_activity": "unknown",
    "lat": DEFAULT_LAT,
    "lon": DEFAULT_LON,
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
    "buffer_size": 128,
    "overlap": 64,
    "lat": DEFAULT_LAT,
    "lon": DEFAULT_LON,
    "sample_count": 0,
    "last_prediction_time": 0,
    "prediction_cooldown": 1.0,
}

# 数据采集模式状态（标注数据收集）
COLLECTED_DIR = Path(__file__).parent.parent / "data" / "collected"
COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

collection_state = {
    "active": False,
    "current_activity": None,
    "sample_counts": {"walking": 0, "stationary": 0, "running": 0, "jumping": 0},
    "buffer": [],
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
        "activities": ["walking", "stationary", "running", "jumping"],
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

    activity_sequence = data.get("activity_sequence", ["walking", "stationary", "running", "jumping"]) if data else ["walking", "stationary", "running", "jumping"]
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
                        # 静止检测：加速度方差太小 → 没有周期性运动，步频归零
                        if np.var(acc_mag) < 0.3:
                            stream_state["step_freq"] = 0
                        else:
                            n = len(acc_mag)
                            fft_vals = np.abs(fft(acc_mag))[:n // 2]
                            freqs = fftfreq(n, 1 / 50)[:n // 2]
                            mask = (freqs >= 0.5) & (freqs <= 5.0)
                            if np.any(mask):
                                peak = np.max(fft_vals[mask])
                                noise = np.mean(fft_vals[mask])
                                # 主频必须明显高于噪声基线才有效
                                if peak > noise * 2.0:
                                    stream_state["step_freq"] = float(freqs[mask][np.argmax(fft_vals[mask])])
                                else:
                                    stream_state["step_freq"] = 0
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
            # 静止检测：加速度方差太小 → 没有周期性运动，步频归零
            if np.var(acc_mag_vals) >= 0.3:
                n = len(acc_mag_vals)
                fft_vals = np.abs(fft(acc_mag_vals))[:n // 2]
                freqs = fftfreq(n, 1 / 50)[:n // 2]
                mask = (freqs >= 0.5) & (freqs <= 5.0)
                if np.any(mask):
                    peak = np.max(fft_vals[mask])
                    noise = np.mean(fft_vals[mask])
                    # 主频必须明显高于噪声基线才有效
                    if peak > noise * 2.0:
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


# ====== 数据采集模式：手机端标注数据收集 ======

@socketio.on("start_collection")
def handle_start_collection(data):
    """开始采集某个运动类型的标注数据"""
    activity = data.get("activity")
    if activity not in ["walking", "stationary", "running", "jumping"]:
        emit("error_msg", {"message": "无效的运动类型"})
        return

    collection_state["active"] = True
    collection_state["current_activity"] = activity
    collection_state["buffer"] = []

    print(f"[Collect] 开始采集: {activity}")
    socketio.emit("status", {"message": f"手机端正在采集: {activity}"})
    emit("collection_started", {"activity": activity})


@socketio.on("collect_sensor_data")
def handle_collect_sensor_data(data):
    """接收带活动标签的传感器原始数据并缓存"""
    if not collection_state["active"]:
        return

    activity = collection_state["current_activity"]
    try:
        sample = {
            "timestamp": data.get("timestamp", time.time()),
            "acc_x": float(data.get("acc_x", 0)),
            "acc_y": float(data.get("acc_y", 0)),
            "acc_z": float(data.get("acc_z", 0)),
            "gyro_x": float(data.get("gyro_x", 0)),
            "gyro_y": float(data.get("gyro_y", 0)),
            "gyro_z": float(data.get("gyro_z", 0)),
            "label": activity,
        }
    except (ValueError, TypeError) as e:
        print(f"[Collect] 数据格式错误: {e}")
        return

    collection_state["buffer"].append(sample)
    collection_state["sample_counts"][activity] += 1

    # 每 500 个点推送一次计数给手机端更新 UI
    total = collection_state["sample_counts"][activity]
    if total % 500 == 0:
        emit("collection_progress", {
            "activity": activity,
            "count": total,
            "counts": collection_state["sample_counts"],
        })


@socketio.on("stop_collection")
def handle_stop_collection():
    """停止当前采集，把缓存数据写入 CSV"""
    if not collection_state["active"] or collection_state["current_activity"] is None:
        emit("error_msg", {"message": "当前没有进行中的采集"})
        return

    activity = collection_state["current_activity"]
    count = len(collection_state["buffer"])

    if count > 0:
        df = pd.DataFrame(collection_state["buffer"])
        filepath = COLLECTED_DIR / f"{activity}.csv"

        # 追加模式：如果之前采过同类型，拼在一起
        if filepath.exists():
            existing = pd.read_csv(filepath)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(filepath, index=False)
        print(f"[Collect] {activity} 保存 {count} 条 → {filepath.name}（累计 {len(df)} 条）")
    else:
        print(f"[Collect] {activity} 本次无数据，跳过保存")

    collection_state["active"] = False
    collection_state["current_activity"] = None
    collection_state["buffer"] = []

    socketio.emit("status", {"message": f"手机端 {activity} 采集完成，共 {collection_state['sample_counts'][activity]} 条"})
    emit("collection_stopped", {
        "activity": activity,
        "counts": collection_state["sample_counts"],
    })


@socketio.on("get_collection_status")
def handle_get_collection_status():
    """查询各活动已采集的数据量"""
    emit("collection_status", {
        "counts": collection_state["sample_counts"],
        "active": collection_state["active"],
        "current_activity": collection_state["current_activity"],
    })


def _train_on_collected_data():
    """用采集到的真实数据重新训练模型（后台线程）"""
    from preprocessing import SensorDataProcessor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib as jl

    socketio.emit("status", {"message": "开始训练模型..."})

    try:
        # 1. 加载所有采集的 CSV 文件
        all_dfs = []
        sample_id = 0
        for csv_file in sorted(COLLECTED_DIR.glob("*.csv")):
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            df["sample_id"] = f"collected_{sample_id:03d}"
            all_dfs.append(df)
            sample_id += 1

        if not all_dfs:
            socketio.emit("status", {"message": "错误：没有采集到任何数据"})
            socketio.emit("train_result", {"success": False, "error": "没有采集到数据"})
            return

        raw_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[Train] 加载 {len(raw_df)} 条原始数据，来自 {len(all_dfs)} 个 CSV 文件")

        # 2. 预处理：滤波 + 滑窗特征提取
        processor_local = SensorDataProcessor(low_cut=0.5, high_cut=20.0)
        features_df = processor_local.process_dataset(raw_df)
        print(f"[Train] 特征提取完成：{features_df.shape}")

        # 3. 准备训练数据
        X = features_df.drop(columns=["label", "window_start", "window_end", "sample_id"], errors="ignore")
        y = features_df["label"]

        # 保存特征名（覆盖旧的）
        feature_names_new = list(X.columns)

        # 标准化
        scaler_new = StandardScaler()
        X_scaled = scaler_new.fit_transform(X)

        # 标签编码
        le_new = LabelEncoder()
        y_enc = le_new.fit_transform(y)

        # 4. 训练
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        clf.fit(X_scaled, y_enc)
        train_acc = clf.score(X_scaled, y_enc)
        print(f"[Train] Random Forest 训练完成，训练集准确率: {train_acc:.2%}")

        # 5. 更新全局模型
        global classifier, scaler, label_encoder, feature_names, processor
        classifier = clf
        scaler = scaler_new
        label_encoder = le_new
        feature_names = feature_names_new
        processor = processor_local

        # 6. 保存模型
        jl.dump(classifier, MODEL_DIR / "classifier.pkl")
        jl.dump(scaler, MODEL_DIR / "scaler.pkl")
        jl.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
        jl.dump(feature_names, MODEL_DIR / "feature_names.pkl")

        # 保存采集数据到 raw 目录（方便查看）
        raw_df.to_csv(DATA_DIR / "raw" / "collected_raw.csv", index=False)
        features_df.to_csv(DATA_DIR / "processed" / "collected_features.csv", index=False)

        activity_counts = raw_df["label"].value_counts().to_dict()
        msg = f"训练完成！训练集准确率 {train_acc:.2%}，共 {len(X)} 个窗口"
        print(f"[Train] {msg}")
        socketio.emit("status", {"message": msg})
        socketio.emit("train_result", {
            "success": True,
            "accuracy": round(train_acc, 4),
            "windows": len(X),
            "features": len(feature_names_new),
            "activity_counts": activity_counts,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        socketio.emit("status", {"message": f"训练失败: {str(e)}"})
        socketio.emit("train_result", {"success": False, "error": str(e)})


@socketio.on("train_model")
def handle_train_model():
    """手机端触发：用采集数据训练模型"""
    if classifier is not None:
        emit("warning", {"message": "这将覆盖当前模型，确定要重新训练吗？"})

    thread = threading.Thread(target=_train_on_collected_data, daemon=True)
    thread.start()
    emit("status", {"message": "训练已在后台启动..."})


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
