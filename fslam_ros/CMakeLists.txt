cmake_minimum_required(VERSION 3.5)
project(fslam_ros)

# Find and import ament_cmake first
find_package(ament_cmake REQUIRED)
# Now we can use ament_cmake functions
ament_package_xml()

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++17 -march=native -Wno-deprecated -Wno-deprecated-declarations"
)
add_definitions("-DENABLE_SSE")

# Set paths
set(HARDCODED_COLCON_DIR ~/colcon_ws) # Hardcoded path (replace with your path containing ) [pavan]
set(FSLAM_PATH ${HARDCODED_COLCON_DIR}/src/FSLAM)
set(FSLAM_LIBRARY_PATH ${HARDCODED_COLCON_DIR}/src/FSLAM/build/lib)
set(THIRD_PARTY_PATH ${FSLAM_PATH}/Thirdparty/CompiledLibs)

# Set CMAKE_MODULE_PATH to include the directory with custom Find modules
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

set(CMAKE_PREFIX_PATH 
    ${CMAKE_PREFIX_PATH}
    ${THIRD_PARTY_PATH}/lib/cmake
)

# Find dependencies
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4.9.0 REQUIRED PATHS ${THIRD_PARTY_PATH}/lib/cmake/opencv4)
find_package(Pangolin 0.2 REQUIRED PATHS ${THIRD_PARTY_PATH}/lib/cmake/Pangolin)
find_package(DBoW3 REQUIRED PATHS ${THIRD_PARTY_PATH}/lib/cmake/DBoW3)

# Use custom Find modules
find_package(SuiteParse REQUIRED MODULE)
find_package(Eigen3 REQUIRED MODULE)
find_package(G2O REQUIRED)

find_package(Boost COMPONENTS system thread REQUIRED)

# Find FSLAM library and headers
if (FSLAM_PATH)
    message("FSLAM_PATH set, trying to find library.")
    message("---- EXPECTING FSLAM sources at\n\"${FSLAM_PATH}\"")
    set(FSLAM_INCLUDE_DIRS 
        "${FSLAM_PATH}/src"
        "${FSLAM_PATH}/Thirdparty/Sophus"
        "${FSLAM_PATH}/Thirdparty/g2o"
        "${FSLAM_PATH}/Thirdparty/DBow3"
        "${FSLAM_PATH}/Thirdparty/opencv-4.9.0"
    )
    message("---- LOOKING FOR FSLAM library at\n\"${FSLAM_LIBRARY_PATH}\"")
    find_library(FSLAM_LIBRARY HSLAM_lib ${FSLAM_LIBRARY_PATH})
else()
    message(FATAL_ERROR "FSLAM_PATH not set. Please set FSLAM_PATH.")
endif()

if (NOT FSLAM_INCLUDE_DIRS)
    message(FATAL_ERROR "FSLAM headers not found and/or FSLAM_PATH not set.")
endif()
if (NOT FSLAM_LIBRARY)
    message(FATAL_ERROR "FSLAM library not found and/or FSLAM_LIBRARY_PATH not set.")
endif()

message("---- Found FSLAM library at \"${FSLAM_LIBRARY}\"")

if (G2O_INCLUDE_DIR)
    set(G2O_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
endif()


# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${FSLAM_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
    ${DBoW3_INCLUDE_DIRS}
    ${SUITEPARSE_INCLUDE_DIRS}
    ${THIRD_PARTY_PATH}/include
    ${CHOLMOD_INCLUDE_DIR}
)

# Declare a C++ executable
add_executable(fslam_live src/main.cpp)

# Specify libraries to link against
target_link_libraries(fslam_live
    ${FSLAM_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBS}
    ${G2O_LIBS}
    ${DBoW3_LIB_DIR}/libDBoW3.so
    ${SUITEPARSE_LIBRARIES}
    cxsparse
    cholmod
)

# Link ROS2 dependencies
ament_target_dependencies(fslam_live
    rclcpp
    sensor_msgs
    std_srvs
    geometry_msgs
    cv_bridge
)

# Install
install(TARGETS fslam_live
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()