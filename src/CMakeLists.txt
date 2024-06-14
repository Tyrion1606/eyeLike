cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR "C:/Users/Davi/Downloads/opencv/build")
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 11)

include_directories(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE( eyeLike main.cpp findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h constants.h helpers.cpp helpers.h)

TARGET_LINK_LIBRARIES( eyeLike ${OpenCV_LIBS} )
