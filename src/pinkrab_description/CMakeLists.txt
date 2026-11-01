cmake_minimum_required(VERSION 3.10)
project(pinkrab_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes mjcf rviz config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
