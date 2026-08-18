#include "ppg_pipeline.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip> 
#include <fstream> 
#include <sstream> 

#ifdef _WIN32
#include <windows.h>
#endif

// 读取数据
bool read_csv(const std::string& filename,
            std::vector<double>& input,
            std::vector<double>& ax,
            std::vector<double>& ay,
            std::vector<double>& az) {
  std::ifstream file(filename); 
  if (!file) { 
    std::printf("ERROR: cannot open %s\n", filename);
    return false;
  }
  std::string line;
  std::getline(file, line); 
  while (std::getline(file, line)) {  
    if (line.empty()) continue;
    double v1,v2,v3,v4;
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf", &v1, &v2, &v3, &v4) == 4) {
      input.emplace_back(v1),ax.emplace_back(v2);
      ay.emplace_back(v3),az.emplace_back(v4);
    }  
  }
  return true;
}

// 输出数据1
void write_three_columns(const std::vector<double>& id,
                      const std::vector<double>& hr_mean,
                      const std::vector<double>& hr_median,
                    //  const std::vector<double>& expected,
                      const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
      std::cerr << "无法打开文件: " << filename << std::endl;
      return;
  }
  file << "id,hr_mean,hr_median\n";
  // file << "id_time,hr_mean,hr_median\n";
  file << std::fixed << std::setprecision(3); 
  size_t n = id.size();
  for (size_t i = 0; i < n; ++i) {
      file << id[i] << "," << hr_mean[i] << "," << hr_median[i] << "\n";
  }
  file.close();
}

// 将 Scene 枚举转换为字符串
std::string scene_to_string(PPGPipelineAll::Scene scene) {
    switch (scene) {
        case PPGPipelineAll::Scene::Rest:   return "Rest";
        case PPGPipelineAll::Scene::Motion: return "Motion";
        default:                            return "Unknown";
    }
}

  // 输出数据2
void write_columns(const std::vector<PPGPipelineAll::Segment>& segments,
                   const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    file << "start,end,scene\n";
    for (const auto& seg : segments) {
        file << seg.start << "," << seg.end << ","
             << scene_to_string(seg.scene) << "\n";
    }
    file.close();
}

int main(){
  #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // 关键：让 Windows 控制台用 UTF-8 解码
  #endif

  const char* inp_path  = "ppg.csv";
  const char* out_path = "ppg_result.csv";

  std::vector<double> ppg,ax,ay,az;
  if (!read_csv(inp_path, ppg,ax,ay,az)) return 1;

  // scipy 导出的系数：bandpass 0.5-4Hz, order=3, fs=128
  std::vector<double> b = {
    0.000537684437389, 0.000000000000000, -0.001613053312167,
        0.000000000000000, 0.001613053312167, 0.000000000000000,
        -0.000537684437389
  };
  std::vector<double> a = {
    1.000000000000000, -5.643142570463496, 13.288878707777238,
        -16.716372702839404, 11.847588878674351, -4.485854579814458,
        0.708902361483485
  };

  const double samples = 128;
  const double window_sec = 2;
  const double threshold = 109;
  const int window_size = 2;

  PPGPipelineAll PPGPA(samples,window_sec,threshold,window_size,0.5,0.1,0.1,16,0.1,1e-10,b,a);

  // 运动或静息时候的参数
  struct Config {
    // nlms参数
    double mu;
    std::size_t filt_order;
    // 峰值检测参数
    int min_distance;
    double ratio_threshold;
    double ratio_prominence;
    // int welch_nperseg;
    // double kf_Q;
    // double kf_R;
  };
  const Config rest_config = {0.05, 16, static_cast<int>(0.5*samples), 0.5, 0.5};
  const Config motion_config = {0.1,16, static_cast<int>(0.3*samples), 0.5, 0.3};

  std::vector<PPGPipelineAll::Segment> segments;
  segments = PPGPA.segment(ppg,ax,ay,az);

  std::vector<double> id;
  std::vector<double> all_hr_mean;
  std::vector<double> all_hr_median;

  // const char* out_path_1 = "ppg_nlms_10_out_tem.csv";
  // write_columns(segments,out_path_1);

  for (const auto& [seg_start, seg_end, scene] : segments){
    std::vector<double> signal,ax_tem,ay_tem,az_tem;
    signal.assign(ppg.begin() + seg_start, ppg.begin() + seg_end);
    ax_tem.assign(ax.begin() + seg_start, ax.begin() + seg_end);
    ay_tem.assign(ay.begin() + seg_start, ay.begin() + seg_end);
    az_tem.assign(az.begin() + seg_start, az.begin() + seg_end);
    int seg_len = seg_end - seg_start;
    // 太短跳过
    if (seg_len < samples * 2) {
      continue;
    }  
    double HALF_SEGMENT = (seg_start+seg_end)/2;
    // 去直流
    std::vector<double> signal_dc =  PPGPA.remove_dcc(signal);

    // NLMS运动伪影去除
    const Config& cfg = (scene == PPGPipelineAll::Scene::Rest) ? rest_config : motion_config;
    PPGPA.set_mu(cfg.mu);
    PPGPA.set_filt_order(cfg.filt_order);
    PPGPA.set_min_distance(cfg.min_distance);
    PPGPA.set_ratio_threshold(cfg.ratio_threshold);
    PPGPA.set_ratio_prominence(cfg.ratio_prominence);
    std::vector<std::vector<double>> refs = {ax_tem,ay_tem,az_tem};
    std::vector<double> signal_nlms =  PPGPA.nlms_process(signal_dc,refs);
    
    size_t n = signal_nlms.size();
    std::vector<double> signal_filt(n);
    // 带通滤波
    PPGPA.filtfilt(signal_nlms, signal_filt); 

    // 移动平均
    std::vector<double> signal_mr = PPGPA.movingrange(signal_filt);

    // z标准化
    std::vector<double> signal_zscore = PPGPA.z_score(signal_mr);

    // 峰值检测
    std::vector<int> signal_peaks = PPGPA.findpeaks(signal_zscore);

    // 计算心率
    // PPGPipeline::HR_Result hr_result = PPGP.cal_hr(signal_peaks);
    auto hr_result = PPGPA.cal_hr(signal_peaks);
    double hr_mean = hr_result.hr_mean;
    double hr_median = hr_result.hr_median;

    id.push_back(HALF_SEGMENT); 
    all_hr_mean.push_back(hr_mean); 
    all_hr_median.push_back(hr_median); 
  }
  write_three_columns(id,all_hr_mean,all_hr_median,out_path);
  return 1;
}

