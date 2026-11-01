cmake_minimum_required(VERSION 3.10)
project(pinkrab_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/TeleopState.msg
  msg/TorqueState.msg
  srv/Estop.srv
  srv/SetTeleopMode.srv
  srv/SetTorque.srv
  DEPENDENCIES builtin_interfaces
)

ament_package()
