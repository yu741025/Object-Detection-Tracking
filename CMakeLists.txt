cmake_minimum_required(VERSION 3.10)
project(object_detection_tracking)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到 OpenCV 包
find_package(OpenCV REQUIRED)

# 添加可執行文件
add_executable(${PROJECT_NAME} src/main.cpp)

# 連接 OpenCV 庫
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# 複製必要的文件到構建目錄
file(COPY ${CMAKE_SOURCE_DIR}/yolov4.cfg
     DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/yolov4.weights
     DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/coco.names
     DESTINATION ${CMAKE_BINARY_DIR})