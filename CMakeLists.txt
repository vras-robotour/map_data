cmake_minimum_required(VERSION 3.8.2)
project(path_planning)

find_package(catkin REQUIRED COMPONENTS message_generation geometry_msgs std_msgs)

catkin_package(CATKIN_DEPENDS
  message_runtime
  geometry_msgs
  std_msgs 
)
