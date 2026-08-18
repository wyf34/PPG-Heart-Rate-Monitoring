#pragma once
#include <vector>
#include <cstddef> 

class PPGPipelineAll{
  private:
    double samples_;
    double window_sec_;   // 运动强度滑窗长度(秒)
    double threshold_;     // 运动判定阈值: std(acc_mag) > threshold → 运动

    std::vector<double> b_;
    std::vector<double> a_;
    int window_size_;
    int min_distance_; 
    double ratio_threshold_;
    double ratio_prominence_;

    // 运动伪影部分
    std::size_t filt_order_;
    double mu_;
    double delta_; 
    struct Channel{
      std::vector<double> weights; // 自适应权重系数
      std::vector<double> x_buffer; // 历史样本
    };
    std::vector<Channel> channels_; // 多通道
    // 滤波
    std::vector<double> forward_filter(const std::vector<double>& input) const;

  public:
    PPGPipelineAll(double samples,double window_sec,double threshold,int window_size,int min_distance,double ratio_threshold,
      double ratio_prominence,std::size_t filt_order, double mu,double delta,
      const std::vector<double>& b,const std::vector<double>& a);

    // ACC预处理
    enum class Scene { Rest, Motion }; // 场景类型
    struct Segment {
        int    start;   // 段起始索引(含)
        int    end;     // 段结束索引(不含)
        Scene  scene;   // 运动 or 静息
    };
    std::vector<Segment> segment(const std::vector<double>& input,
      const std::vector<double>& ax,const std::vector<double>& ay, const std::vector<double>& az);
    
    // 去直流
    std::vector<double> remove_dcc(const std::vector<double>& input);

    // NLMS去伪影
    std::vector<double> nlms_process(const std::vector<double>& input,const std::vector<std::vector<double>>& references);

    // 带通滤波
    void filtfilt(const std::vector<double>& input_signal, std::vector<double>& output_signal); 

    // 移动平均
    std::vector<double> movingrange(const std::vector<double>& input);

    // z标准化
    std::vector<double> z_score(const std::vector<double>& signal);

    // 归一化
    std::vector<double> min_max(const std::vector<double>& signal);

    // 峰值检测
    std::vector<int> findpeaks(const std::vector<double>& input);

    // 计算心率值
    struct HR_Result {
    double hr_mean;
    double hr_median;
    };

    HR_Result cal_hr(const std::vector<int>& signal_peaks);

    double get_samples() const { return samples_; }
    int get_window_size() const { return window_size_; }
    int get_min_distance() const { return min_distance_; }
    double get_ratio_threshold() const { return ratio_threshold_; }
    double get_ratio_prominence() const { return ratio_prominence_; }
    std::size_t get_filt_order() const {return filt_order_;};
    double get_mu() const {return mu_;};
    const std::vector<Channel>& get_channels() const {return channels_;};
    const std::vector<double>& get_b() const { return b_; }
    const std::vector<double>& get_a() const { return a_; }

    void set_samples(double samples){samples_ = samples;};
    void set_window_sec(double window_sec){window_sec_  = window_sec;};
    void set_threshold(double threshold){threshold_  = threshold;};
    void set_window_size(double window_size){window_size_  = window_size;};
    void set_min_distance(double min_distance){min_distance_  = min_distance;};
    void set_ratio_threshold(double ratio_threshold){ratio_threshold_  = ratio_threshold;};
    void set_ratio_prominence(double ratio_prominence){ratio_prominence_  = ratio_prominence;};
    void set_filt_order(double filt_order){filt_order_  = filt_order;};
    void set_mu(double mu){mu_  = mu;};
    void set_delta(double delta){delta_  = delta;};
    void set_b(const std::vector<double>& new_b){b_ = new_b;};
    void set_a(const std::vector<double>& new_a){a_ = new_a;};

};


