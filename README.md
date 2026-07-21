# 运动数据分析与展示系统

端到端的实时步态分析系统：手机浏览器采集 IMU 传感器数据 → 服务器特征提取与模型预测 → 电脑仪表盘实时可视化。

## 功能特性

- **📱 手机端标注数据采集**：浏览器 DeviceMotion API 零成本采集，四种运动（走路/静止/跑步/跳跃），数据自动保存 CSV
- **🧠 闭环模型训练**：手机采集完四项后一键触发训练，服务器后台执行 → 替换模型 → 即刻验证
- **📡 实时预测**：手机 WebSocket 推送传感器数据到服务器，~50Hz 采样，128 点窗口推理，1s 更新一次结果
- **📊 电脑仪表盘**：Chart.js 实时波形图 + Leaflet 运动轨迹地图 + 运动识别面板
- **🔌 Socket.IO 双向通信**：心跳保活、断连检测、多客户端广播

## 技术栈

| 类别 | 技术 |
|------|------|
| 数据采集 | DeviceMotion API + Geolocation API（手机浏览器） |
| 数据处理 | Python, NumPy, Pandas, SciPy |
| 机器学习 | Scikit-learn (Random Forest, SVM) |
| Web 后端 | Flask + Flask-SocketIO |
| Web 前端 | HTML5 + Chart.js + Leaflet.js + 高德地图瓦片 |
| 实时通信 | Socket.IO（WebSocket） |
| 模型持久化 | joblib |

## 项目结构

```
motion-analysis-system/
├── data/
│   ├── raw/                   # 原始数据（模拟: imu_data.csv 等）
│   ├── processed/             # 特征提取后的数据
│   └── collected/             # 手机端采集的真实数据（walking.csv, running.csv, ...）
├── models/                    # 训练好的模型文件（classifier.pkl, scaler.pkl 等）
├── src/
│   ├── data_generator.py      # 模拟数据生成（NumPy 信号仿真）
│   ├── preprocessing.py       # 巴特沃斯滤波 + 时频域特征提取
│   ├── train_model.py         # Random Forest / SVM 模型训练
│   └── app.py                 # Flask Web 应用 + Socket.IO 事件处理
├── static/
│   ├── css/style.css
│   ├── js/dashboard.js        # 仪表盘前端逻辑
│   └── figures/               # 混淆矩阵、特征重要性图
├── templates/
│   ├── index.html             # 电脑端仪表盘
│   └── mobile.html            # 手机端采集 + 预测页面
└── requirements.txt
```

## 快速开始

### 1. 环境准备

```bash
cd motion-analysis-system
pip install -r requirements.txt
```

### 2. 启动服务

```bash
cd src
python app.py
```

### 3. 访问

- **电脑仪表盘**：`http://localhost:5000` — 查看实时波形、轨迹、识别结果
- **手机采集页**：`http://<电脑IP>:5000/mobile` — 采集数据或实时预测（同 WiFi）

## 使用指南

### 方式一：真实数据采集 + 训练 + 预测（推荐）

用你自己的手机数据训练模型，告别模拟数据。

```
手机访问 /mobile
  → 按 🚶走路 → 走路 10~30 秒 → 点 ⏹停止
  → 按 🛑静止 → 保持不动 10 秒  → 点 ⏹停止
  → 按 🏃跑步 → 跑步 10~30 秒 → 点 ⏹停止
  → 按 ⤴跳跃 → 跳跃 10~30 秒 → 点 ⏹停止
  → 四个全亮 ✅→ 点 ⚡训练模型
  → 等待训练完成 → 点 📡实时预测 → 看电脑仪表盘
```

### 方式二：模拟数据 + 预测（快速体验）

用 NumPy 生成的模拟信号直接训练模型。

```bash
cd src
python data_generator.py     # 生成 60 个模拟样本
python preprocessing.py      # 滤波 + 特征提取
python train_model.py        # 训练模型

python app.py                # 启动服务
# 仪表盘点"开始实时数据流"即可看到演示效果
```

### Chrome 传感器配置

手机用 HTTP（非 HTTPS）访问时，Chrome 默认禁用传感器。配置方法：

```
chrome://flags → 搜索 unsafely-treat-insecure-origin-as-secure
  → 填入 http://<电脑IP>:5000 → Enabled → 点 Relaunch
```

## 关键技术

### 步频计算

通过 FFT 提取加速度量级的主频率，并增加两层校验：

- **方差阈值**：`np.var(acc_mag) < 0.3` 判定为静止，步频归零
- **信噪比校验**：主频峰值需大于噪声均值 × 2，确保不是噪声干扰

### 特征工程

8 个传感器通道 × 19 个特征 = 152 维特征向量：

- 8 通道：acc_x/y/z、gyro_x/y/z、acc_magnitude、gyro_magnitude
- 时域 12 个：均值、标准差、最大/小值、范围、RMS、偏度、峰度、能量、峰峰值、过零率、SMA
- 频域 7 个：主频、频谱质心、频谱能量、4 频带能量分布

### Socket.IO 事件一览

| 事件名 | 方向 | 频率 | 用途 |
|--------|------|------|------|
| `start_collection` | 📱→🖥 | 按需 | 开始标注采集某活动 |
| `collect_sensor_data` | 📱→🖥 | ~50 Hz | 带标签的传感器原始数据 |
| `stop_collection` | 📱→🖥 | 按需 | 保存 buffer → CSV |
| `train_model` | 📱→🖥 | 按需 | 用采集数据重新训练模型 |
| `mobile_connect` | 📱→🖥 | 连接时 | 手机进入实时预测模式 |
| `mobile_sensor_data` | 📱→🖥 | ~50 Hz | 实时预测传感器数据 |
| `mobile_disconnect` | 📱→🖥 | 断开时 | 清理状态 |
| `start_stream` | 💻→🖥 | 按需 | 启动模拟数据流 |
| `sensor_data` | 🖥→💻 | ~10 Hz | 波形数据 → Chart.js |
| `activity_update` | 🖥→💻 | ~1 Hz | 运动类型 + 置信度 + 步频 |
| `trajectory_update` | 🖥→💻 | ~1 Hz | GPS → Leaflet 地图 |
| `server_heartbeat` | 🖥→📱 | 2s | 心跳保活 |

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 电脑端仪表盘 |
| `/mobile` | GET | 手机端采集页面 |
| `/api/status` | GET | 系统状态（模型是否加载、活动列表） |
| `/api/dataset/info` | GET | 数据集统计信息 |
| `/api/dataset/sample/<id>` | GET | 单个样本的 IMU 数据 |
| `/api/model/metrics` | GET | 模型评估指标 |

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 手机传感器无数据 | Chrome HTTP 禁止 DeviceMotion | 配置 unsafely-treat 白名单 |
| 仪表盘"等待手机连接" | 未点 📡实时预测 | 手机端点"开始实时预测" |
| 地图白屏/灰屏 | OpenStreetMap 被墙 | 已切换高德瓦片，无需处理 |
| 静止时步频显示 1 Hz | 早期版本 FFT 对噪声误判 | 已修复：方差阈值 + 信噪比 |
| 训练后识别不准 | 采集数据量太少或动作不标准 | 每类至少采集 1000+ 条，动作幅度到位 |
