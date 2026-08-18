#include "ppg_pipeline.h"
#include <iostream>
#include <stdexcept>
#include <cmath> 
#include <algorithm> 
#include <numeric>
#include <vector>

PPGPipelineAll::PPGPipelineAll(double samples,double window_sec,double threshold,int window_size,int min_distance,double ratio_threshold,
      double ratio_prominence,size_t filt_order, double mu,double delta,
      const std::vector<double>& b,const std::vector<double>& a)
      :samples_(samples),window_sec_(window_sec),threshold_(threshold),window_size_(window_size),min_distance_(min_distance),ratio_threshold_(ratio_threshold),
      ratio_prominence_(ratio_prominence),filt_order_(filt_order),mu_(mu),delta_(delta),
      b_(b),a_(a){
      }

// 去直流
std::vector<double> PPGPipelineAll::remove_dcc(const std::vector<double>& signal){
  if(signal.empty()) return{};
  size_t n = signal.size(); 
  double signal_sum = std::accumulate(signal.begin(),signal.end(),0.0);
  double signal_mean = signal_sum / static_cast<double>(n);

  std::vector<double> out_signal(n);
  for(size_t i = 0;i <= n;++i ){
    out_signal[i] = signal[i] - signal_mean;
  }
  return out_signal;
}


// ACC预处理
std::vector<PPGPipelineAll::Segment> PPGPipelineAll::segment(const std::vector<double>& input,
      const std::vector<double>& ax,const std::vector<double>& ay,
      const std::vector<double>& az){
    const int n = static_cast<int>(std::min({ax.size(), ay.size(), az.size()}));
    std::vector<PPGPipelineAll::Segment> segments;
    std::cout << n << std::endl;
    if (n == 0) return segments;

    // ACC模值
    //（1）计算运动强度（标准差）
    std::vector<double> acc_mag(n);
    for (int i = 0; i < n; ++i) {
        acc_mag[i] = std::sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]);
    }
    std::vector<double> signal_std(n);
    std::vector<char> is_motion_raw(n);
    int min_duration = static_cast<int>(window_sec_ * samples_);
    const int half   = min_duration / 2; 
    for (int i = 0; i < n; ++i) {
      int start = std::max(0, i - half);
      int end   = std::min(n, i + half);  
      int cnt   = end - start;
      if (cnt <= 0) { signal_std[i] = 0.0; continue; }
      std::vector<double> signal_tem(acc_mag.begin() + start, acc_mag.begin() + end);
      size_t tem_num = signal_tem.size();
      double signal_sum = accumulate(signal_tem.begin(),signal_tem.end(),0.0);
      double signal_mean = signal_sum / static_cast<double>(tem_num);
      auto signal_sqsum = accumulate(signal_tem.begin(),signal_tem.end(),0.0,
            [](double acc,double v){return acc + v*v;});
      signal_std[i] = sqrt(signal_sqsum / static_cast<double>(tem_num) - signal_mean*signal_mean);
      // （2）阈值判断
      is_motion_raw[i] = signal_std[i] > threshold_ ? 1 : 0;
      
    }
    //（3）去抖动：连续 window_sec_*samples_ 个点才切换
    int i = 0;
    while (i < n) {
        char current_state = is_motion_raw[i];
        int  j = i + 1;
        while (j < n && is_motion_raw[j] == current_state) j++;   // [i, j) 连续同状态
        if (j - i < min_duration && i > 0) {
            // 太短, 改成前一段的状态
            for (int k = i; k < j; ++k) is_motion_raw[k] = is_motion_raw[i - 1];
        }
        i = j;
    }
    // (4)获取运动或静息列表
    char current_state = is_motion_raw[0];
    int  start = 0;
    for (int i = 1; i < n; ++i) {
        if (is_motion_raw[i] != current_state) {
            segments.push_back({start, i, current_state ? Scene::Motion : Scene::Rest});
            start = i;
            current_state = is_motion_raw[i];
        }
    }
    // 最后一段 [start, n)处理
    segments.push_back({start, n, current_state ? Scene::Motion : Scene::Rest});
    return segments;
}

// NLMS去伪影
std::vector<double> PPGPipelineAll::nlms_process(const std::vector<double>& input,
    const std::vector<std::vector<double>>& references){
    const std::size_t n_samples = input.size();
    const std::size_t n_refs = references.size();

    channels_.assign(n_refs,Channel{});
    for(std::size_t i = 0;i < n_refs;++i){
      channels_[i].weights.assign(filt_order_,0.0);
      channels_[i].x_buffer.assign(filt_order_,0.0);
    }

    std::vector<double> output(n_samples);
    for(std::size_t i = 0;i < n_samples;++i){
      
      // 移位处理
      for(std::size_t j = 0;j < n_refs;++j){
        std::vector<double>& buf = channels_[j].x_buffer;
        std::copy_backward(buf.begin(),buf.end()-1,buf.end());
        buf[0] = references[j][i];
      }
      
      // 计算各个通道下的加速度相关值之和
      double estimated_noise = 0;
      for(std::size_t j = 0;j < n_refs;++j){
        const std::vector<double>& w = channels_[j].weights;
        const std::vector<double>& xb = channels_[j].x_buffer;
        double channel_output = 0.0;
        for(std::size_t n = 0;n < filt_order_; ++n){
          channel_output += w[n] * xb[n];
        }
        estimated_noise += channel_output;
      }

      double e_n = input[i] - estimated_noise;

      // 更新权重weights
      for(std::size_t j = 0;j < n_refs;++j){
        std::vector<double>& w = channels_[j].weights;
        const std::vector<double>& xb = channels_[j].x_buffer;
        double input_power = 1e-10;
        for(std::size_t n = 0;n < filt_order_; ++n){
          input_power += xb[n] * xb[n];
        }
        const double scale = mu_ * e_n / input_power;
        for(std::size_t n = 0;n < filt_order_; ++n){
          w[n] += scale * xb[n];
        }
      }
      output[i] = e_n;
    }

  return output;
}

// 带通滤波
std::vector<double> PPGPipelineAll::forward_filter(const std::vector<double>& input) const {
  int n = input.size();
  int nb = b_.size();
  int na = a_.size();
  std::vector<double> output(n);
  double sum_b = 0.0,sum_a = 0.0;
  for(double v : b_) sum_b += v;
  for(double v : a_) sum_a += v;
  double G = sum_b / sum_a;  
  double x0 = input.empty() ? 0.0 : input[0];
  // 初始化历史缓冲区
  std::vector<double> x_hist(nb,x0);
  std::vector<double> y_hist(na > 1 ? na-1 : 0,G * x0);
  for (int i = 0;i < n;i++){
    // 右移 x 历史，新样本放最前
    for(int j = nb - 1;j > 0; j--){
      x_hist[j] = x_hist[j-1];
    }
    x_hist[0] = input[i];
    double y = 0.0;
    for(int j = 0;j < nb;j++){
      y += b_[j] * x_hist[j];
    }
    for(int j = 1;j < na;j++){
      y -= a_[j] * y_hist[j-1];
    }
    //右移 y 历史，新输出放最前
    for(int j = na -2;j > 0;j--){
      y_hist[j] = y_hist[j-1];
    }
    if(na > 1){
      y_hist[0] = y;
    }
    output[i] = y;
  }
  return output;
}

// 计算滤波信号
void PPGPipelineAll::filtfilt(const std::vector<double>& input_signal,std::vector<double>& output_signal){
  // filtfilt = 正向滤波 + 反向滤波，消除相位延迟
  int n = input_signal.size();
  int nb = b_.size();
  int na = a_.size();

// scipy filtfilt 的 padding 长度: 3 * max(len(a), len(b))
  int padlen = 3 * (nb > na ? nb : na);
//  信号太短就不做padding
  if (n <= padlen){
    std::vector<double> fwd = forward_filter(input_signal);
    std::vector<double> rev(fwd.rbegin(), fwd.rend());
    std::vector<double> filt_again = forward_filter(rev);
    output_signal.assign(filt_again.rbegin(),filt_again.rend()); // 将 filt_again 反转后存入 output_signal
    return ;
  }
  // 边界填充(消除起始/结束瞬态),左右两边采用奇反射，使填充信号在边界处平滑过渡，避免滤波器产生瞬态振荡。
  // 左边： padded[i] = 2*x[0] - x[padlen -i]
  // 右边： padded[n+padlen+i] = 2*x[n-1] - x[n-2-i]
  std::vector<double> padded(n + 2 * padlen);
  for (int i = 0;i < padlen;i++){
    padded[i] = 2.0 * input_signal[0] - input_signal[padlen - i];
  }
  for (int i = 0;i < n;i++){
    padded[padlen+i] = input_signal[i];
  }
  for (int i = 0;i < padlen;i++){
    padded[padlen + n + i] = 2.0 * input_signal[n - 1] - input_signal[n - 2 - i];
  }

  // 2.正向滤波
  std::vector<double> fwd = forward_filter(padded);

  // 3.反转
  std::vector<double> rev(fwd.rbegin(),fwd.rend());

  // 4.对反转信号再做一次滤波
  std::vector<double> filt_again = forward_filter(rev);

  // 5.反转回来
  std::vector<double> result(filt_again.rbegin(),filt_again.rend());

  // 6.去掉两段padding
  output_signal.assign(result.begin() + padlen,result.begin() + padlen + n);
}

// 移动平均
std::vector<double> PPGPipelineAll:: movingrange(const std::vector<double>& input) {
  size_t n = input.size();
  std::vector<double> output(n);

  double sum = 0.0;
  int current_index = 0;
  std::vector<double> buffer(window_size_,0.0);

  for(size_t i = 0;i < n;i++){
    sum -= buffer[current_index];
    buffer[current_index] = input[i];
    sum += buffer[current_index];

    // 更新索引
    current_index = (current_index + 1) % window_size_;
    static int count = 0;
    if(count < window_size_) count++;
    output[i] = sum / static_cast<double>(count);

  };
  return output;
}

// 标准化
std::vector<double> PPGPipelineAll::z_score(const std::vector<double>& signal){
  if(signal.empty()) return{};
  size_t n = signal.size();
  double signal_sum = accumulate(signal.begin(),signal.end(),0.0);
  double signal_mean = signal_sum / static_cast<double>(n);

  auto signal_sqsum = accumulate(signal.begin(),signal.end(),0.0,
        [](double acc,double v){return acc + v*v;});
  double signal_std = sqrt(signal_sqsum / static_cast<double>(n) - signal_mean*signal_mean);

  if(signal_std < 1e-12){
    // 信号近乎恒定 → 全填 0
    return std::vector<double>(n,0.0);
  }
  std::vector<double> normlized(n);
  transform(signal.begin(),signal.end(),normlized.begin(),
          [signal_mean,signal_std](double x){return (x - signal_mean)/signal_std;});
  return normlized;
}

//  归一化
std::vector<double> PPGPipelineAll::min_max(const std::vector<double>& signal){
   if(signal.empty()) return{};
   size_t n = signal.size();
   auto [min_it, max_it] = std::minmax_element(signal.begin(), signal.end());
   double min_val = *min_it;
   double max_val = *max_it;
   double range = max_val - min_val;
   const double eps = 1e-12; 
   if(range < eps){
    return std::vector<double>(n,0.0);
   }

   std::vector<double> out_signal(n);
   for(size_t i = 0;i <= n;++i){
      out_signal[i] = (signal[i] - min_val) / range ;
   }
   return out_signal;
}

// 峰值检测
std::vector<int> PPGPipelineAll::findpeaks(const std::vector<double>& input){
  size_t n = input.size();
  if(n < 3) return{};
  float ppg_sum = accumulate(input.begin(),input.end(),0.0f);
  float ppg_mean = ppg_sum / n;
  auto ppg_sqsum = accumulate(input.begin(),input.end(),0.0f,
        [](float acc,float v){return acc + v*v;});
  float ppg_std = sqrt(ppg_sqsum / n - ppg_mean*ppg_mean);
  double height = ratio_threshold_ * ppg_std; 
  std::vector<int> candidates;
  // 找局部最大值 + 幅度阈值
  for(int i = 1;i < n-1;i++){
    if(input[i] > input[i-1] && input[i] > input[i+1]){
      if(input[i] > height){
         candidates.push_back(i);
      }
    }
  }
  // 距离约束
  std::vector<int> peaks;
  for(int canl : candidates){
    if(peaks.empty()){
      peaks.push_back(canl);
    }else{
      int last_peak = peaks.back();
      if(canl - last_peak < min_distance_){
         if(input[canl] > input[last_peak]){
          peaks.back() = canl;
         } 
      }else{
        peaks.push_back(canl);
      }
    }
  }

  double prominence = ratio_prominence_ * ppg_std; // 峰值的显著性，即峰值相对于周围谷地的垂直距离
  std::vector<int> filtered;
  for (int peak : peaks) {
    double peak_height = input[peak];
    int left_higher = 0; 
    for (int i = peak - 1; i >= 0; i--) {
        if (input[i] > peak_height) {
            left_higher = i;
            break;
        }
    }
    int right_higher = n - 1;
    for (int i = peak + 1; i < n; i++) {
        if (input[i] > peak_height) {
            right_higher = i;
            break;
        }
    }
    double min_val = peak_height;
    for (int i = left_higher; i <= right_higher; i++) {
        if (input[i] < min_val) min_val = input[i];
    }
    double dist_prom = peak_height - min_val;
    if(dist_prom > prominence){
      filtered.push_back(peak);
    }
  }
  return filtered;
}

//计算心率值
PPGPipelineAll::HR_Result PPGPipelineAll::cal_hr(const std::vector<int>& signal_peaks){
  // 计算心率值
  if (signal_peaks.size() < 2){
    return {0.0, 0.0};
  }
  //计算 RR 间期（秒）
  std::vector<double> rr_intervals;
  rr_intervals.reserve(signal_peaks.size() - 1);
  for (size_t i = 1; i < signal_peaks.size(); ++i) {
      double interval_sec = (signal_peaks[i] - signal_peaks[i - 1]) / samples_;
      rr_intervals.push_back(interval_sec);
  }
  // 转换为 BPM
  std::vector<double> heart_rates;
  heart_rates.reserve(rr_intervals.size());
  for (double rr : rr_intervals) {
      heart_rates.push_back(60.0 / rr);      
  }
  // 均值
  double sum = std::accumulate(heart_rates.begin(), heart_rates.end(), 0.0);
  double mean = sum / heart_rates.size();
  // 中位数
  std::vector<double> sorted = heart_rates;
  std::sort(sorted.begin(), sorted.end());
  double median;
  size_t n = sorted.size();
  if (n % 2 == 0) {
      median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
  } else {
      median = sorted[n / 2];
  }
  double hr_median = median;
  double hr_mean = mean;
  return {hr_mean,hr_median};
}





