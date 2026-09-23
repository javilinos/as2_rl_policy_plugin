cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME as2_rl_policy)
project(${PROJECT_NAME})

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === ROS 2 dependencies =====================================================
set(PROJECT_DEPENDENCIES
  ament_cmake
  rclcpp
  pluginlib
  as2_core
  as2_msgs
  as2_motion_controller
  geometry_msgs
  sensor_msgs
  std_msgs
  diagnostic_msgs
)

foreach(DEPENDENCY ${PROJECT_DEPENDENCIES})
  find_package(${DEPENDENCY} REQUIRED)
endforeach()

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# === Core library (no ROS) ===================================================
add_library(${PROJECT_NAME}_core SHARED
  src/core/angles.cpp
  src/core/course.cpp
  src/core/decode.cpp
  src/core/gates.cpp
  src/core/mission.cpp
  src/core/observation.cpp
  src/core/policy.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_core PUBLIC Eigen3::Eigen PRIVATE yaml-cpp)

# === Plugin library ==========================================================
add_library(${PROJECT_NAME}_lib SHARED
  src/rl_policy.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_lib ${PROJECT_NAME}_core)
ament_target_dependencies(${PROJECT_NAME}_lib ${PROJECT_DEPENDENCIES})

# === Install =================================================================
install(DIRECTORY include/ DESTINATION include)

install(TARGETS ${PROJECT_NAME}_core ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY tests/fixtures DESTINATION share/${PROJECT_NAME}/tests)
install(PROGRAMS scripts/engage.py DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core ${PROJECT_NAME}_lib)
ament_export_dependencies(${PROJECT_DEPENDENCIES} Eigen3 yaml-cpp)

pluginlib_export_plugin_description_file(as2_motion_controller plugins.xml)

# === Tests ===================================================================
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

ament_package()
