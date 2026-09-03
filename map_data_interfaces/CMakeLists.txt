cmake_minimum_required(VERSION 3.8)
project(map_data_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PlanRoute.action"
  DEPENDENCIES action_msgs geographic_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
