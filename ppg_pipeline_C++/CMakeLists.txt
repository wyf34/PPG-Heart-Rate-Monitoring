cmake_minimum_required(VERSION 3.10)          # 1. 最低 CMake 版本

project(ppg_pipeline VERSION 1.0 LANGUAGES CXX)  # 2. 项目名+语言

set(CMAKE_CXX_STANDARD 17)                    # 3. C++ 标准: 17
set(CMAKE_CXX_STANDARD_REQUIRED ON)           #    必须用 17, 没有就报错

add_executable(                                # 4. 生成可执行文件
    ppg_pipeline_test                           #    输出名 (等价 g++ 的 -o)
    ppg_pipeline.cpp                                   #    源文件清单
    ppg_pipeline_main.cpp                      #    (等价 g++ 后面列的所有 .cpp)
)
