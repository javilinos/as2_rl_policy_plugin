find_package(ament_cmake_gtest REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(FIXTURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# Core tests: the core library alone, no ROS.
set(CORE_TESTS
  policy
  observation
  decode
  gates
  mission
)

foreach(TEST_NAME ${CORE_TESTS})
  ament_add_gtest(${PROJECT_NAME}_${TEST_NAME}_gtest ${TEST_NAME}_gtest.cpp)
  target_link_libraries(${PROJECT_NAME}_${TEST_NAME}_gtest ${PROJECT_NAME}_core yaml-cpp)
  target_compile_definitions(${PROJECT_NAME}_${TEST_NAME}_gtest PRIVATE
    RL_POLICY_FIXTURES_DIR="${FIXTURES_DIR}")
endforeach()

# Adapter tests: the plugin on an as2 node, and inside the controller manager.
foreach(TEST_NAME rl_policy rl_policy_manager)
  ament_add_gtest(${PROJECT_NAME}_${TEST_NAME}_gtest ${TEST_NAME}_gtest.cpp
    SKIP_LINKING_MAIN_LIBRARIES
    TIMEOUT 120)
  target_link_libraries(${PROJECT_NAME}_${TEST_NAME}_gtest ${PROJECT_NAME}_lib yaml-cpp)
  ament_target_dependencies(${PROJECT_NAME}_${TEST_NAME}_gtest
    ${PROJECT_DEPENDENCIES} ament_index_cpp rosgraph_msgs tf2_ros)
  target_compile_definitions(${PROJECT_NAME}_${TEST_NAME}_gtest PRIVATE
    RL_POLICY_FIXTURES_DIR="${FIXTURES_DIR}"
    RL_POLICY_CONFIG_DIR="${PROJECT_SOURCE_DIR}/config")
endforeach()
