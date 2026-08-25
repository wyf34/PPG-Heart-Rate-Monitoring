# PPG-Heart-Rate-Monitoring
## 结合ACC数据对PPG信号进行处理计算心率
结合 PPG 与三轴加速度计信号，通过 NLMS 自适应滤波去除运动伪影、SQI 信号质量评估、动静场景自适应参数，实现多场景心率检测。Python 主实现 + C++ 移植版。
### 文件介绍
- ppg_pipeline.py为数据处理程序
- ppg_pipeline_C++为C++版数据处理程序
### 数据情况
使用智能手环采集3个多小时手腕PPG和加速度计数据，采样率均为128Hz，其中加速度计数据为3轴（X、Y、Z）数据。每30s给定一个标准心率值。
### 特性
- **NLMS 多通道自适应滤波**：三轴 ACC 作参考，per-channel 归一化，权重跨窗口流式保持
- **0.1Hz 高通 + 0.5-4Hz 带通**：去基线漂移 + 保留心率频段（filtfilt 零相位）
- **SQI 三指标质量评估**：峰度 / SNR / 周期性，权重 0.2/0.4/0.4，SNR 一票否决
- **ACC 动静判断分场景参数**：mu / peak_distance / prominence 按动静自适应
- **滑窗 SQI + 多数投票**：6s 物理窗口取中心 4s，1s 步进，3 窗口投票平滑状态
### 算法pipeline
```
PPG + 三轴ACC
  → 0.1Hz高通滤波(去直流、去除极端基线漂移)
  → ACC去直流 → 动静判断(ACC模值滑窗std > 阈值)         
  → NLMS自适应滤波去除运动伪影（mu随动静: 运动0.1/静息0.05）
  → 0.5~4Hz的巴特沃斯带通滤波  (filtfilt零相位)
  → 滑窗SQI评估（滑窗计算包含峰度、SNR、周期性指标）
  → 多数投票平滑状态(3窗口)
  → Z-score 归一化
  → 峰值检测 (prominence按SQI, distance按运动/静息场景)
  → 心率计算
  → 输出 hr_result.csv 
```

### 参数说明

