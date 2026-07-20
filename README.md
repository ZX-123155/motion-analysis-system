# 运动数据分析与展示系统

端到端的运动数据分析系统，实现IMU传感器数据采集模拟、步态特征提取、机器学习运动识别分类，以及实时Web可视化展示。

## 功能特性

- **多运动类型模拟数据生成**：走路、跑步、跳跃、高抬腿四种运动的IMU（加速度计+陀螺仪）和GNSS轨迹数据
- **信号处理与特征工程**：巴特沃斯滤波、时域+频域特征提取
- **机器学习分类**：随机森林 (Random Forest) 和 SVM 模型训练与评估
- **实时Web仪表盘**：基于 Flask + WebSocket 的端到端实时展示系统

## 技术栈

| 类别 | 技术 |
|------|------|
| 数据处理 | Python, NumPy, Pandas, SciPy |
| 机器学习 | Scikit-learn (Random Forest, SVM) |
| Web后端 | Flask, Flask-SocketIO |
| Web前端 | HTML5, CSS3, JavaScript, Chart.js, Leaflet.js |
| 实时通信 | WebSocket |
| 可视化 | Matplotlib, Chart.js |

## 项目结构

```
motion-analysis-system/
├── data/
│   ├── raw/                  # 原始模拟数据
│   └── processed/            # 处理后的特征数据
├── models/                   # 训练好的模型文件
├── src/
│   ├── data_generator.py     # 模拟数据生成
│   ├── preprocessing.py      # 数据预处理与特征提取
│   ├── train_model.py        # 模型训练与评估
│   └── app.py                # Flask Web 应用
├── static/
│   ├── css/style.css         # 样式表
│   ├── js/dashboard.js       # 前端仪表盘逻辑
│   └── figures/              # 生成的图表
├── templates/
│   └── index.html            # 主页面
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
cd motion-analysis-system
pip install -r requirements.txt
```

### 2. 生成数据并训练模型

```bash
cd src

# 生成模拟数据
python data_generator.py

# 数据预处理与特征提取
python preprocessing.py

# 训练分类模型
python train_model.py
```

### 3. 启动Web应用

```bash
cd src
python app.py
```

浏览器访问 `http://localhost:5000`

### 4. 使用实时仪表盘

1. 打开网页后，点击"开始实时数据流"按钮
2. 系统会模拟手机传感器实时传输数据
3. 左侧面板实时显示识别出的运动类型和置信度
4. 中间面板动态绘制传感器加速度/角速度波形
5. 右侧地图实时更新运动轨迹
6. 可通过下拉菜单选择特定运动类型或全部类型

## 模型性能

训练完成后，模型评估指标可在 `/api/model/metrics` 接口查看，混淆矩阵和特征重要性图在 `static/figures/` 目录中。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/status` | GET | 系统状态 |
| `/api/dataset/info` | GET | 数据集信息 |
| `/api/dataset/sample/<id>` | GET | 样本数据 |
| `/api/model/metrics` | GET | 模型评估指标 |

## WebSocket 事件

| 事件 | 方向 | 说明 |
|------|------|------|
| `start_stream` | 客户端→服务器 | 启动数据流 |
| `stop_stream` | 客户端→服务器 | 停止数据流 |
| `sensor_data` | 服务器→客户端 | 传感器数据 |
| `trajectory_update` | 服务器→客户端 | 轨迹更新 |
| `activity_update` | 服务器→客户端 | 运动识别结果 |
