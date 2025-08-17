cmake_minimum_required(VERSION 3.8)
project(explore)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(map_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
set(DEPENDENCIES
  rclcpp
  std_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  nav2_msgs
  nav_msgs
  map_msgs
  nav2_costmap_2d
  visualization_msgs
)
include_directories(
    include
)
install(
  DIRECTORY include/explore/
  DESTINATION include/explore/
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/aruco_boundary_node.py
  src/aruco_detector_node.py
  src/twist_to_stamp.py
  src/aruco_detect_test.py
  DESTINATION lib/${PROJECT_NAME}
)
add_executable(explore
  src/costmap_client.cpp
  src/explore.cpp
  src/frontier_search.cpp
)
target_include_directories(explore PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(explore ${rclcpp_LIBRARIES})

ament_target_dependencies(explore ${DEPENDENCIES})

install(TARGETS explore
  DESTINATION lib/${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

ament_package()