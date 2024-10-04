cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(export_pth)

find_package(Torch REQUIRED)

add_executable(export_pth export_pth.cpp)
target_link_libraries(export_pth "${TORCH_LIBRARIES}")
set_property(TARGET export_pth PROPERTY CXX_STANDARD 14)