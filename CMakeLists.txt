cmake_minimum_required(VERSION 3.8.2)
project(map_data)

find_package(catkin REQUIRED COMPONENTS rospy geodesy)

catkin_package(CATKIN_DEPENDS
  message_runtime
  geodesy
)
