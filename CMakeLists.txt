cmake_minimum_required(VERSION 3.18)
project(chrysalis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(include)

add_executable(chrysalis src/main.cu)
