# Stage 1: Base image with Ubuntu 22.04 and ROS2 Humble
FROM osrf/ros:humble-desktop-jammy AS base
ARG DEBIAN_FRONTEND=noninteractive

# Install colcon build tool dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions

# Stage 2: Additional dependencies and Realsense SDK
#FROM base AS dependencies
# Install Camera Realsense D435 dependencies
RUN apt-get update && apt-get install -y ros-humble-usb-cam
#Install system wide dependencies
RUN apt-get install -y \
    libgl1-mes-dev \
    libglew-dev \
    libsuitesparse-dev \
    libeigen3-dev \
    libboost-all-dev \
    cmake \
    build-essential \
    git \
    libzip-dev \
    ccache \
    freeglut3-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    ninja-build
#Install pangolin gui dependencies (optional)
RUN apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libavdevice-dev
#Install additional libraries
RUN apt-get install -y \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libv4l-dev \
    libgtk2.0-dev \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools\
    v4l-utils\
    wget \
    unzip

# Set the working directory
WORKDIR /colcon_ws/src/FSLAM/Thirdparty
# Copy Thirdparty folder to the container
COPY Thirdparty /colcon_ws/src/FSLAM/Thirdparty
#Download additional thirdparty libraries
ARG cvVersion=4.9.0
ARG DL_opencv="https://github.com/opencv/opencv/archive/${cvVersion}.zip"
ARG DL_contrib="https://github.com/opencv/opencv_contrib/archive/${cvVersion}.zip"
RUN wget ceres-solver.org/ceres-solver-1.14.0.tar.gz \
    && tar -zxf ceres-solver-1.14.0.tar.gz \
    && wget -O opencv.zip -nc "${DL_opencv}" \
    && unzip opencv.zip \
    && rm opencv.zip \
    && cd opencv-4.9.0 \
    && wget -O opencv_contrib.zip -nc "${DL_contrib}" \
    && unzip opencv_contrib.zip \
    && rm opencv_contrib.zip
#build Thirparty libraries using script(cmake)
RUN chmod +x build.sh && ./build.sh
#Copy project files
WORKDIR /colcon_ws/src/FSLAM
COPY FSLAM /colcon_ws/src/FSLAM
#build FSLAM project using cmake
RUN mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RelwithDebInfo && make -j 10
#copy fslam_ros wrapper and build using colcon
COPY fslam_ros /colcon_ws/src/fslam_ros
WORKDIR /colcon_ws/
RUN colcon build --packages-select fslam_ros

#Source colcon and specify the entry point of the container    
RUN sed --in-place --expression \
      '$isource "/colcon_ws/install/setup.bash"' \
      /ros_entrypoint.sh
#Copy calibration files
WORKDIR /colcon_ws/src/FSLAM
COPY res /colcon_ws/src/res
WORKDIR /colcon_ws
CMD ["bash"]
