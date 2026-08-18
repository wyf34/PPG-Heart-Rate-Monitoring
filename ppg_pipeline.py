import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks,welch
import matplotlib
matplotlib.use('Agg')
from scipy.stats import kurtosis as scipy_kurtosis
from collections import Counter
import pywt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 动静状态判断
def judge_motion_rest(ppg, fs, accel_x, accel_y, accel_z, window_sec=2, threshold=0.5, filter_order=16, mu=0.1):
    """
    Args:
        ppg: PPG信号
        fs: 采样频率，此处ppg和acc采样频率一致
        accel_x, accel_y, accel_z: 三轴加速度计信号
        window_sec: 动静滑窗阈值，默认2s，需考虑生理过渡；
        threshold：动静判断阈值，根据ACC标准差数据判断；
        filter_order：NLMS使用阶数；每个通道的滤波器阶数；
        mu：NLMS使用步长，运动0.1，静息0.05；
    Returns:
        filtered_ppg: 去除运动伪影后的PPG信号；
        accel_x_dc,accel_y_dc,accel_z_dc：去直流后的ACC数据，方便后续信号质量评估时调用
    """
    # 去直流
    accel_x_dc = accel_x - np.mean(accel_x)
    accel_y_dc = accel_y - np.mean(accel_y)
    accel_z_dc = accel_z - np.mean(accel_z)
    accel_x_tem = accel_x_dc.copy()
    accel_y_tem = accel_y_dc.copy()
    accel_z_tem = accel_z_dc.copy()

    # 可选的0.5-20Hz带通滤波，保留可能包含运动伪影的频带
    #     nyquist = 0.5 * self.fs
    #     low = 0.5 / nyquist
    #     high = 20.0 / nyquist
    #     b, a = butter(2, [low, high], btype='band')
    #     accel_x_filt = filtfilt(b, a, accel_x_dc)
    #     accel_y_filt = filtfilt(b, a, accel_y_dc)
    #     accel_z_filt = filtfilt(b, a, accel_z_dc)
    #     accel_x_tem = accel_x_filt.copy()
    #     accel_y_tem = accel_y_filt.copy()
    #     accel_z_tem = accel_z_filt.copy()

    # 做动静判断
    # ACC 运动检测
    accel_mag = np.sqrt(accel_x_tem ** 2 + accel_y_tem ** 2 + accel_z_tem ** 2)
    motion_intensity = pd.Series(accel_mag).rolling(window_sec * fs).std()
    motion_judge = (motion_intensity > threshold).astype(int)  # 1为运动，0为静息

    # NLMS算法
    n_samples = len(ppg)
    ppg = np.asarray(ppg).flatten()
    accel_x_cal = np.asarray(accel_x_tem[:n_samples]).flatten()
    accel_y_cal = np.asarray(accel_y_tem[:n_samples]).flatten()
    accel_z_cal = np.asarray(accel_z_tem[:n_samples]).flatten()

    # 组合所有参考信号
    references = [accel_x_cal, accel_y_cal, accel_z_cal]  # 把三个轴的加速度计信号打包成列表，方便后续循环处理。
    n_refs = len(references)
    # 初始化
    weights = np.zeros((n_refs, filter_order))  # 每个参考信号一个滤波器, # shape (3, 16)
    filtered_ppg = []
    # 信号去直流
    processed_signal = ppg - np.mean(ppg)
    # 0.1Hz高通滤波去除极端基线漂移
    nyquist = 0.5 * fs
    high_cut = 0.1 / nyquist
    # 使用二阶巴特沃兹高通滤波器
    b, a = butter(2, high_cut, btype='high')
    ppg_highpassed = filtfilt(b, a, processed_signal)
    # 为每个参考信号创建缓冲区
    x_buffers = [np.zeros(filter_order) for _ in range(n_refs)]
    for n in range(n_samples):
        # 更新mu值
        if motion_judge[n] > 0:
            mu = 0.1
        else:
            mu = 0.05
        for i in range(n_refs):
            x_buffers[i][1:] = x_buffers[i][:-1]
            x_buffers[i][0] = references[i][n]
        estimated_noise = 0
        input_power_total = 0
        for i in range(n_refs):
            channel_output = np.dot(weights[i], x_buffers[i])
            estimated_noise += channel_output  # 运动伪影
            input_power_total += np.dot(x_buffers[i], x_buffers[i])
        e_n = ppg_highpassed[n] - estimated_noise # 计算实际信号
        # 更新所有滤波器的权重
        for i in range(n_refs):
            input_power = np.dot(x_buffers[i], x_buffers[i]) + 1e-10  # 平方和
            weights[i] = weights[i] + (mu * e_n / input_power) * x_buffers[i]
        filtered_ppg.append(e_n)
    return filtered_ppg,accel_x_dc,accel_y_dc,accel_z_dc

# 巴特沃斯带通滤波器
def butter_bandpass_filter(input_signal, fs,lowcut=0.5, highcut=4.0, order=4): # 0.5~4Hz的带通滤波
    """
    巴特沃斯带通滤波器
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    output_signal = filtfilt(b, a, input_signal)  # 双向滤波避免相位失真
    return output_signal

# SQI指标打分函数
# 峰度 → 0-1 分
def _kurtosis_to_score(kurt):
    """
    理想区间 2.5-5.0 → 1.0
    边缘 1.5-2.5 或 5.0-8.0 → 线性衰减
    差 <1.5 或 >8.0 → 0
    """
    if 2.5 <= kurt <= 5.0:
        return 1.0
    elif 1.5 <= kurt < 2.5:
        return (kurt - 1.5) / (2.5 - 1.5)
    elif 5.0 < kurt <= 8.0:
        return 1.0 - (kurt - 5.0) / (8.0 - 5.0)
    else:
        return 0.0
# SNR → 0-1 分
def _snr_to_score(snr_db):
    """
    >10 dB → 1.0（信号远强于噪声）
    0-10 dB → 线性
    <0 dB → 0（噪声盖过信号）
    """
    if snr_db >= 10:
        return 1.0
    elif snr_db >= 0:
        return snr_db / 10.0
    else:
        return 0.0
# 周期性（归一化自相关峰值）→ 0-1 分
def _periodicity_to_score(periodicity):
    """
    >0.7 → 1.0（强周期，干净PPG）
    0.3-0.7 → 线性
    <0.3 → 0（无周期性，噪声/严重伪影）
    """
    if periodicity >= 0.7:
        return 1.0
    elif periodicity >= 0.3:
        return (periodicity - 0.3) / (0.7 - 0.3)
    else:
        return 0.0

# SQI 指标计算
def compute_sqi_with_acc(ppg_signal, acc_x, acc_y, acc_z, fs=128):
    """
    三个指标：
      ① 峰度 kurtosis —— PPG 波形有尖峰（脉搏波），理想~3
      ② SNR —— 心率带功率 / 噪声带功率
      ③ 周期性 —— 自相关函数在心率周期的峰值
    """
    # 1. 三个指标计算
    signal = np.array(ppg_signal, dtype=float)
    n = len(signal)
    # 去均值（SQI 在去 DC 后评估）
    signal = signal - np.mean(signal)
    diff_signal = np.abs(np.diff(signal))
    flat_ratio = np.mean(diff_signal < 1e-6)
    is_saturated = flat_ratio > 0.1  # 超过 10% 的点是一样的

    # 峰度计算
    kurt = scipy_kurtosis(signal, fisher=False)
    kurt_score = _kurtosis_to_score(kurt)

    # SNR
    # 信号带：0.5-4 Hz（心率 + 谐波）
    # 噪声带：4-8 Hz（高频噪声/运动伪影）
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, n))
    sig_mask = (freqs >= 0.5) & (freqs <= 4.0)
    noise_mask = (freqs > 4.0) & (freqs <= 8.0)
    sig_power = np.sum(psd[sig_mask])
    noise_power = np.sum(psd[noise_mask])
    snr_db = 10 * np.log10(sig_power / (noise_power + 1e-10))
    snr_score = _snr_to_score(snr_db)

    # 周期性
    # 在 0.5-2.0 秒（30-120bpm 对应周期）找最大值。PPG 心率周期在 0.5-2.0 秒之间。如果这段有明显自相关峰，说明信号有心率节律；如果没有，说明噪声/伪影主导。
    autocorr = np.correlate(signal, signal, 'full')[n - 1:]  # 只取正 lag
    autocorr = autocorr / autocorr[0]  # 归一化（lag=0 处为1）
    # 在 0.5-2.0 秒（30-120bpm 对应周期）范围内找最大峰
    min_lag = int(0.5 * fs)  # 0.5s = 120bpm
    max_lag = int(2.0 * fs)  # 2.0s = 30bpm
    search_range = autocorr[min_lag:max_lag + 1]
    if len(search_range) > 0:
        periodicity = np.max(search_range)
    else:
        periodicity = 0.0
    periodicity_score = _periodicity_to_score(periodicity)

    # 综合计算权重：SNR (0.4) + 周期性(0.4) + 峰度(0.2)
    # 但 SNR 有一票否决权：SNR 太低时信号一定不好，SNR 否决：SNR < 0.2 时（<2dB），不管其他指标多高，总分不超过 0.5
    base_score = 0.2 * kurt_score + 0.4 * snr_score + 0.4 * periodicity_score
    if snr_score < 0.2:
        total_score = min(base_score, 0.5)  # 封顶在 marginal
    else:
        total_score = base_score
    # 饱和直接判 bad（不管其他指标多高）
    if is_saturated:
        total_score = 0.0
        # label = 'bad (saturated)'
        label = 'bad'
    elif total_score >= 0.6: # 这里的阈值需要根据实际数据调整
        label = 'good'
    elif total_score >= 0.3:
        label = 'marginal'
    else:
        label = 'bad'

    # 2. ACC 运动检测
    accel_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    motion = np.std(accel_mag)
    motion_detected = 0
    if motion > 0.05:  # 检测到运动，0.05为经验值，可根据传感器灵敏度调。
        motion_detected = 1

    return {
        'score': round(total_score, 3),
        'label': label,
        'motion_detected': motion_detected,
        'saturated': is_saturated,
        'kurtosis': round(kurt, 2),
        'kurtosis_score': round(kurt_score, 3),
        'snr_db': round(snr_db, 2),
        'snr_score': round(snr_score, 3),
        'periodicity': round(periodicity, 3),
        'periodicity_score': round(periodicity_score, 3),
    }

def sliding_window_sqi(ppg, fs, acc_x, acc_y, acc_z, window_sec_sqi=4, buffer_sec=1, step_sec=1):
    """
    滑窗评估SQI：扩展窗口 + 中心取值
    Args:
        ppg: PPG信号
        fs: 采样频率，此处ppg和acc采样频率一致
        accel_x, accel_y, accel_z: 三轴加速度计信号
        window_sec_sqi: 逻辑窗口（取峰区），扩展窗口为 window_sec + 2*buffer_sec = 6秒
        buffer_sec: 前后缓冲区
        step_sec：滑窗步长
    Returns:
        sqi_results：信号质量评估结果
        timestamps:  被评估信号对应的时间
    """
    physical = int((window_sec_sqi + 2 * buffer_sec) * fs)  # 6秒=768点
    step = int(step_sec * fs)
    n = len(ppg)
    sqi_results = []
    timestamps = []
    # 中心对齐滑窗
    for center in range(physical // 2, n - physical // 2, step):
        phys_start = center - physical // 2
        phys_end = phys_start + physical
        seg = ppg[phys_start:phys_end]
        seg_acc_x = acc_x[phys_start:phys_end]
        seg_acc_y = acc_y[phys_start:phys_end]
        seg_acc_z = acc_z[phys_start:phys_end]
        sqi = compute_sqi_with_acc(seg, seg_acc_x, seg_acc_y, seg_acc_z, fs)
        sqi['start_idx'] = phys_start
        sqi['end_idx'] = phys_end
        sqi['center_time'] = center / fs  # 秒
        sqi_results.append(sqi)
        timestamps.append(center / fs)
    return sqi_results, timestamps

if __name__ == '__main__':
    fs = 128  # ppg采样率，Hz，此处ppg和acc采样率一致
    data = np.load(r'data.npy',allow_pickle=True).item()
    phase0_data = data['phase 0']
    ppg = phase0_data['PPG wrist']
    accel_x = phase0_data['IMU X wrist']
    accel_y = phase0_data['IMU Y wrist']
    accel_z = phase0_data['IMU Z wrist']

    #     data = pd.read_csv(r'ppg_nlms_10.csv')
    #     ppg = data['sigal']
    #     accel_x = data['ax']
    #     accel_y = data['ay']
    #     accel_z = data['az']

    # 运动伪影处理
    window_sec = 2; threshold = 0.5; filter_order = 16; mu = 0.1
    ppg_nlms, ax_dc, ay_dc, az_dc = judge_motion_rest(ppg, fs, accel_x, accel_y, accel_z, window_sec=2, threshold=0.5,
                                                      filter_order=16, mu=0.1)
    # 0.5~4Hz的带通滤波
    lowcut = 0.5; highcut = 4.0; order = 4
    ppg_filter = butter_bandpass_filter(ppg_nlms, fs, lowcut=0.5, highcut=4.0, order=4)
    # 滑窗SQI评估
    window_sec_sqi = 4; buffer_sec = 1; step_sec = 1
    sqi_results, timestamps = sliding_window_sqi(ppg_filter, fs, ax_dc, ay_dc, az_dc, window_sec_sqi=4, buffer_sec=1, step_sec=1)
    # 多数投票（3窗口）平滑状态
    labels = []
    for i in range(len(sqi_results)):
        # 取最近3个窗口
        recent = sqi_results[max(0, i - 2):i + 1]
        recent_labels = [r['label'].split(' ')[0] for r in recent]
        # 多数投票，取众数
        vote = Counter(recent_labels).most_common(1)[0][0]
        labels.append(vote)

    # 计算心率
    heart_rate_timeline = []  # (time, hr, confidence)
    valid_peaks_all = []  # (time,peaks)
    peaks_ori = []
    logical = int(4 * fs)  # 4秒=512点
    buffer = int(1 * fs)  # 1秒=128点
    for i, (sqi, ts) in enumerate(zip(sqi_results, timestamps)):
        label = labels[i]
        if label == 'bad':
            # 不输出心率
            heart_rate_timeline.append((ts, None))
            continue
        start = sqi['start_idx']
        end = sqi['end_idx']
        seg_ppg = ppg_filter[start:end]

        # 有ACC数据的情况
        if sqi['motion_detected'] > 0:
            peak_distance = int(0.3 * fs)  # 运动， 小间隔，防高心率漏检
        else:
            peak_distance = int(0.5 * fs)  # 静止，  大间隔，过滤重搏波和呼吸伪影

        if label == 'marginal':
            peak_prominence = 0.5  # 严，只留突出的峰
        else:
            peak_prominence = 0.3  # 松

        # Z标准化
        ppg_zscore = np.round((seg_ppg - np.mean(seg_ppg)) / np.std(seg_ppg) + 1e-10)

        # 峰值检测
        peaks, _ = find_peaks(ppg_zscore,
                              height=0.5,
                              distance=peak_distance,
                              prominence=peak_prominence,
                              width=(6, 50))

        peaks_ori.append((ts, peaks))

        # 只保留中心4秒内的峰（去掉缓冲区）
        logical_start = buffer  # 128
        logical_end = logical + buffer  # 640
        valid_peaks = peaks[(peaks >= logical_start) & (peaks < logical_end)]
        valid_peaks_all.append((ts, valid_peaks))
        if len(valid_peaks) >= 2:
            rr = np.diff(valid_peaks) / fs
            hr = 60.0 / np.mean(rr)
            # 生理合理性检查
            if 30 < hr < 240:
                heart_rate_timeline.append((ts, hr))
            else:
                heart_rate_timeline.append((ts, 250))
        else:
            heart_rate_timeline.append((ts, 999))

    df_out = pd.DataFrame(heart_rate_timeline, columns=['timestamp', 'heart_rate'])
    df_out.to_csv(r'hr_result.csv',index=False)























