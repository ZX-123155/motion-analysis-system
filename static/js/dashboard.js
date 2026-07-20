/**
 * 运动数据分析系统 - 实时仪表盘
 * 功能：WebSocket实时通信、传感器波形图、轨迹地图、运动识别展示
 */

// ====== 全局状态 ======
const state = {
    socket: null,
    chart: null,
    map: null,
    pathLine: null,
    markers: [],
    sensorData: { acc_x: [], acc_y: [], acc_z: [], gyro_x: [], gyro_y: [], gyro_z: [], labels: [] },
    maxDataPoints: 150,
    currentTab: 'acc',
    dataCount: 0,
    chartColors: {
        acc_x: '#e74c3c',  // 红色
        acc_y: '#2ecc71',  // 绿色
        acc_z: '#3498db',  // 蓝色
        gyro_x: '#e67e22',
        gyro_y: '#9b59b6',
        gyro_z: '#1abc9c',
    },
    activityIcons: {
        walking: '\u{1F6B6}',
        running: '\u{1F3C3}',
        jumping: '\u{29D7}',
        high_knees: '\u{1F1F7}',
        unknown: '\u2753',
    },
    activityNames: {
        walking: '走路 Walking',
        running: '跑步 Running',
        jumping: '跳跃 Jumping',
        high_knees: '高抬腿 High Knees',
        unknown: '等待数据...',
    },
};

// ====== 初始化 ======
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initChart();
    connectSocket();
    loadModelMetrics();

    // Tab切换
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentTab = btn.dataset.tab;
            updateChart();
        });
    });
});

// ====== 地图初始化 ======
function initMap() {
    state.map = L.map('map', {
        center: [39.9042, 116.4074],
        zoom: 16,
        zoomControl: true,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
    }).addTo(state.map);

    state.pathLine = L.polyline([], {
        color: '#3498db',
        weight: 4,
        opacity: 0.8,
    }).addTo(state.map);
}

// ====== 图表初始化 ======
function initChart() {
    const ctx = document.getElementById('sensor-chart').getContext('2d');

    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Acc X',
                    data: [],
                    borderColor: state.chartColors.acc_x,
                    backgroundColor: state.chartColors.acc_x + '20',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Acc Y',
                    data: [],
                    borderColor: state.chartColors.acc_y,
                    backgroundColor: state.chartColors.acc_y + '20',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Acc Z',
                    data: [],
                    borderColor: state.chartColors.acc_z,
                    backgroundColor: state.chartColors.acc_z + '20',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 200,
            },
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        font: { size: 11 },
                    },
                },
                tooltip: {
                    enabled: true,
                },
            },
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: '采样点' },
                    ticks: { maxTicksLimit: 10, font: { size: 10 } },
                },
                y: {
                    display: true,
                    title: { display: true, text: '值' },
                    ticks: { font: { size: 10 } },
                },
            },
        },
    });
}

// ====== WebSocket ======
function connectSocket() {
    state.socket = io();

    state.socket.on('connect', () => {
        setStreamStatus(true);
        showToast('已连接到服务器', 'success');
        fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                if (data.model_loaded) {
                    document.getElementById('model-status').textContent = '模型已加载';
                    document.getElementById('model-status').style.color = '#27ae60';
                } else {
                    document.getElementById('model-status').textContent = '模型未加载';
                    document.getElementById('model-status').style.color = '#e74c3c';
                }
            });
    });

    state.socket.on('disconnect', () => {
        setStreamStatus(false);
        showToast('连接断开', 'error');
    });

    state.socket.on('sensor_data', handleSensorData);
    state.socket.on('trajectory_update', handleTrajectoryUpdate);
    state.socket.on('activity_update', handleActivityUpdate);
    state.socket.on('prediction_result', handlePredictionResult);
    state.socket.on('status', (data) => showToast(data.message, 'info'));
    state.socket.on('error', (data) => showToast(data.message, 'error'));
    state.socket.on('warning', (data) => showToast(data.message, 'warning'));
}

// 处理传感器数据
function handleSensorData(data) {
    if (!data || !data.timestamps) return;

    const Ts = data.timestamps.length;
    state.dataCount += Ts;

    // 追加数据
    const append = (arr, newData) => {
        arr.push(...newData);
        while (arr.length > state.maxDataPoints) arr.shift();
    };

    append(state.sensorData.acc_x, data.acc_x);
    append(state.sensorData.acc_y, data.acc_y);
    append(state.sensorData.acc_z, data.acc_z);
    append(state.sensorData.gyro_x, data.gyro_x);
    append(state.sensorData.gyro_y, data.gyro_y);
    append(state.sensorData.gyro_z, data.gyro_z);

    // 更新标签
    const currentLen = state.sensorData.acc_x.length;
    const startIdx = Math.max(0, currentLen - Ts);
    state.sensorData.labels = Array.from({ length: currentLen }, (_, i) => (startIdx + i));

    updateChart();
    document.getElementById('data-counter').textContent = `数据点: ${state.dataCount}`;
}

// 处理轨迹更新
function handleTrajectoryUpdate(data) {
    if (!data || !state.map) return;

    const latlng = [data.lat, data.lon];
    state.pathLine.addLatLng(latlng);
    state.map.panTo(latlng, { animate: true, duration: 0.5 });

    document.getElementById('map-coords').textContent =
        `(${data.lat.toFixed(6)}, ${data.lon.toFixed(6)}) 速度: ${data.speed.toFixed(1)} m/s`;

    // 限制轨迹点数
    const latlngs = state.pathLine.getLatLngs();
    if (latlngs.length > 200) {
        state.pathLine.setLatLngs(latlngs.slice(-200));
    }
}

// 处理运动识别更新
function handleActivityUpdate(data) {
    const act = data.activity || 'unknown';
    const conf = data.confidence || 0;
    const stepFreq = data.step_freq || 0;
    const speed = data.speed || 0;

    // 更新运动类型显示
    const iconEl = document.getElementById('activity-icon');
    const nameEl = document.getElementById('activity-name');
    const barEl = document.getElementById('confidence-bar');
    const textEl = document.getElementById('confidence-text');

    iconEl.innerHTML = state.activityIcons[act] || '\u2753';
    iconEl.className = 'activity-icon active';
    nameEl.textContent = state.activityNames[act] || act;
    barEl.style.width = (conf * 100).toFixed(0) + '%';
    textEl.textContent = `置信度: ${(conf * 100).toFixed(1)}%`;

    // 更新统计
    document.getElementById('stat-step-freq').textContent = stepFreq > 0 ? stepFreq.toFixed(2) : '--';
    document.getElementById('stat-speed').textContent = speed > 0 ? speed.toFixed(2) : '--';
}

// 处理预测结果
function handlePredictionResult(data) {
    handleActivityUpdate({
        activity: data.activity,
        confidence: data.confidence,
    });
}

// ====== 图表更新 ======
function updateChart() {
    if (!state.chart) return;

    const d = state.sensorData;
    const labels = d.labels;
    const tab = state.currentTab;

    if (tab === 'acc') {
        state.chart.data.labels = labels;
        state.chart.data.datasets = [
            { label: 'Acc X (m/s²)', data: d.acc_x, borderColor: state.chartColors.acc_x, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Acc Y (m/s²)', data: d.acc_y, borderColor: state.chartColors.acc_y, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Acc Z (m/s²)', data: d.acc_z, borderColor: state.chartColors.acc_z, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        ];
        state.chart.options.scales.y.title.text = '加速度 (m/s²)';
    } else {
        state.chart.data.labels = labels;
        state.chart.data.datasets = [
            { label: 'Gyro X (rad/s)', data: d.gyro_x, borderColor: state.chartColors.gyro_x, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Gyro Y (rad/s)', data: d.gyro_y, borderColor: state.chartColors.gyro_y, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Gyro Z (rad/s)', data: d.gyro_z, borderColor: state.chartColors.gyro_z, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        ];
        state.chart.options.scales.y.title.text = '角速度 (rad/s)';
    }

    state.chart.update('none');
}

// ====== 控制按钮 ======
function startStream() {
    if (!state.socket || !state.socket.connected) {
        showToast('未连接到服务器', 'error');
        return;
    }

    const select = document.getElementById('activity-select');
    const activities = select.value.split(',');

    state.socket.emit('start_stream', { activity_sequence: activities });

    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    setStreamStatus(true, true);

    // 清空旧数据
    state.sensorData = { acc_x: [], acc_y: [], acc_z: [], gyro_x: [], gyro_y: [], gyro_z: [], labels: [] };
    state.dataCount = 0;
    if (state.pathLine) state.pathLine.setLatLngs([]);
}

function stopStream() {
    if (!state.socket) return;
    state.socket.emit('stop_stream');

    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}

// ====== 辅助函数 ======
function setStreamStatus(connected, streaming = false) {
    const dot = document.getElementById('stream-status');
    const text = document.getElementById('stream-text');

    if (connected && streaming) {
        dot.className = 'status-dot online';
        text.textContent = '数据流运行中';
    } else if (connected) {
        dot.className = 'status-dot online';
        text.textContent = '已连接';
    } else {
        dot.className = 'status-dot offline';
        text.textContent = '未连接';
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

async function loadModelMetrics() {
    try {
        const resp = await fetch('/api/model/metrics');
        if (!resp.ok) return;
        const metrics = await resp.json();

        document.getElementById('model-perf').style.display = 'block';
        const container = document.getElementById('model-metrics');
        container.innerHTML = Object.entries(metrics)
            .map(([k, v]) => `<div class="metric-row"><span>${k}</span><span>${(v * 100).toFixed(1)}%</span></div>`)
            .join('');
    } catch (e) {
        // 静默处理
    }
}

// ====== 窗口事件 ======
window.addEventListener('resize', () => {
    if (state.map) state.map.invalidateSize();
    if (state.chart) state.chart.resize();
});

// 页面卸载时停止数据流
window.addEventListener('beforeunload', () => {
    if (state.socket) state.socket.emit('stop_stream');
});
