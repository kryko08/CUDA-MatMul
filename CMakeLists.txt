cmake_minimum_required(VERSION 3.0)

project(TEMPLATE LANGUAGES CXX CUDA) 


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TEMPALTE main.cu)
