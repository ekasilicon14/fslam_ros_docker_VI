cmake_minimum_required(VERSION 3.5)
project(fslam_ros)
set(HARDCODED_PROJECT_DIR /home/pavanjay/lab) # Hardcoded path to the project directory (replace later: pavan)

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_MODULE_PATH ${HARDCODED_PROJECT_DIR}/colcon_ws/src/fslam_ros/cmake)
set(CMAKE_PREFIX_PATH ${HARDCODED_PROJECT_DIR}/colcon_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4.9.0 REQUIRED PATHS ${HARDCODED_PROJECT_DIR}/colcon_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake/opencv4)
find_package(Pangolin 0.2 REQUIRED PATHS ${HARDCODED_PROJECT_DIR}/colcon_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake)
find_package(DBoW3 REQUIRED)
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)

# Set paths
set(FSLAM_PATH /home/pavanjay/lab/colcon_ws/src/FSLAM)
set(FSLAM_LIBRARY_PATH /home/pavanjay/lab/colcon_ws/src/FSLAM/build/lib)

# Set values from environment variables if not already set
if (NOT FSLAM_PATH)
    set(FSLAM_PATH $ENV{FSLAM_PATH})
endif()
if (NOT FSLAM_LIBRARY_PATH)
    set(FSLAM_LIBRARY_PATH $ENV{FSLAM_LIBRARY_PATH})
    if (NOT FSLAM_LIBRARY_PATH AND FSLAM_PATH)
        set(FSLAM_LIBRARY_PATH "${FSLAM_PATH}/build/lib")
    endif()
endif()

# Find FSLAM library and headers
if (FSLAM_PATH)
    # Manually set path to FSLAM source folder
    message("FSLAM_PATH set, trying to find library.")
    message("---- EXPECTING FSLAM sources at\n\"${FSLAM_PATH}\"")
    set(FSLAM_INCLUDE_DIRS "${FSLAM_PATH}/src" "${FSLAM_PATH}/Thirdparty/Sophus" "${FSLAM_PATH}/Thirdparty/g2o" "${FSLAM_PATH}/Thirdparty/DBow3" "${FSLAM_PATH}/Thirdparty/opencv-4.9.0")
    message("---- LOOKING FOR FSLAM library at\n\"${FSLAM_LIBRARY_PATH}\"")
    find_library(FSLAM_LIBRARY HSLAM_lib ${FSLAM_LIBRARY_PATH})
else()
    message("FSLAM_PATH not set yet, trying to find installed fslam headers and library.")
    find_path(FSLAM_INCLUDE_DIRS FSLAM)
    if (FSLAM_INCLUDE_DIRS)
        set(FSLAM_INCLUDE_DIRS "${FSLAM_INCLUDE_DIRS}/FSLAM")
        message("---- FOUND FSLAM headers at \"${FSLAM_INCLUDE_DIRS}\"")
    endif()
    find_library(FSLAM_LIBRARY fslam)
endif()

if (NOT FSLAM_INCLUDE_DIRS)
    message(FATAL_ERROR "FSLAM headers not found and/or FSLAM_PATH not set.")
endif()
if (NOT FSLAM_LIBRARY)
    message(FATAL_ERROR "FSLAM library not found and/or FSLAM_LIBRARY_PATH not set.")
endif()

message("---- Found FSLAM library at \"${FSLAM_LIBRARY}\"")

# Set additional paths
set(CMAKE_MODULE_PATH /home/pavanjay/lab/colcon_ws/src/fslam_ros/cmake)
set(OpenCV_DIR /home/pavanjay/lab/colcon_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake/opencv4)
set(Pangolin_DIR /home/pavanjay/lab/colcon_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake)
set(DBoW3_DIR /home/pavanjay/lab/colcon_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake/DBoW3)
set(G2O_DIR "/home/pavanjay/lab/colcon_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake/g2o")

# Include directories
include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${FSLAM_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${DBoW3_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
)

# Link directories
link_directories(
  ${G2O_INCLUDE_DIR}
  ${DBoW3_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
)

# Declare a C++ executable
add_executable(fslam_live src/main.cpp)

# Link libraries
target_link_libraries(fslam_live
    ${FSLAM_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    ${DBoW3_LIB_DIR}/libDBoW3.so
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Install
install(TARGETS fslam_live
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()