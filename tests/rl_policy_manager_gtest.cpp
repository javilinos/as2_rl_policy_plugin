#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ament_index_cpp/get_resource.hpp>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "tf2_ros/static_transform_broadcaster.h"
#include "as2_motion_controller/controller_manager.hpp"
#include "as2_motion_controller/testing/mock_platform.hpp"
#include "as2_msgs/msg/control_mode.hpp"
#include "as2_msgs/msg/platform_info.hpp"
#include "as2_msgs/msg/thrust.hpp"
#include "as2_msgs/msg/trajectory_setpoints.hpp"
#include "as2_msgs/srv/set_control_mode.hpp"
#include "fixture_utils.hpp"

namespace
{

constexpr char kNamespace[] = "test_rl_policy_manager";
constexpr char kPolicyFixture[] = "policy_random.yaml";

std::string scoped(const std::string & name)
{
  return std::string("/") + kNamespace + "/" + name;
}

std::shared_ptr<controller_manager::ControllerManager> makeManager(
  const std::vector<rclcpp::Parameter> & extra = {})
{
  const YAML::Node course = rl_policy_test::loadFixture("obs_season2.yaml")["course"];
  const double dt = rl_policy_test::loadFixture(kPolicyFixture)["dt"].as<double>();
  const std::string policy = rl_policy_test::fixturePath(kPolicyFixture);
  const std::string config_dir = RL_POLICY_CONFIG_DIR;

  rclcpp::NodeOptions options;
  options.arguments(
    {"--ros-args", "-r", std::string("__ns:=/") + kNamespace,
      "--params-file",
      ament_index_cpp::get_package_share_directory("as2_motion_controller") +
      "/config/motion_controller_default.yaml",
      "--params-file", config_dir + "/rl_policy_default.yaml"});
  std::vector<rclcpp::Parameter> overrides = {
      rclcpp::Parameter("plugin_name", std::string("rl_policy")),
      rclcpp::Parameter("use_bypass", false),
      rclcpp::Parameter(
        "plugin_available_modes_config_file", config_dir + "/available_modes.yaml"),
      rclcpp::Parameter("cmd_freq", 1.0 / dt),
      rclcpp::Parameter("rl_policy.policies.files", std::vector<std::string>{policy, policy}),
      rclcpp::Parameter("rl_policy.course.gates_x", rl_policy_test::doubles(course["gates_x"])),
      rclcpp::Parameter("rl_policy.course.gates_y", rl_policy_test::doubles(course["gates_y"])),
      rclcpp::Parameter("rl_policy.course.gates_z", rl_policy_test::doubles(course["gates_z"])),
      rclcpp::Parameter("rl_policy.course.gates_yaw", rl_policy_test::doubles(course["gates_yaw"])),
      // The drone does not move here: no gate may time the race out under the test.
      rclcpp::Parameter("rl_policy.race.gate_timeout_s", 600.0),
  };
  for (const rclcpp::Parameter & parameter : extra) {
    overrides.push_back(parameter);
  }
  options.parameter_overrides(overrides);
  return std::make_shared<controller_manager::ControllerManager>(options);
}

// A static drone: its pose on TF, its twist, its rotor speeds and an armed, offboard platform;
// with a course, also a pilot that republishes it on the trajectory reference topic.
class Drone
{
public:
  explicit Drone(
    const as2_msgs::msg::TrajectorySetpoints & course = as2_msgs::msg::TrajectorySetpoints())
  : node_(std::make_shared<rclcpp::Node>(
        "rl_policy_manager_drone", std::string("/") + kNamespace)),
    course_(course)
  {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = node_->now();
    transform.header.frame_id = "earth";
    transform.child_frame_id = std::string(kNamespace) + "/base_link";
    transform.transform.translation.x = 14.5;
    transform.transform.translation.y = 2.0;
    transform.transform.translation.z = 0.2;
    transform.transform.rotation.w = std::cos(0.5 * 3.14);
    transform.transform.rotation.z = std::sin(0.5 * 3.14);
    broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);
    broadcaster_->sendTransform(transform);

    twist_pub_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(
      "self_localization/twist", rclcpp::SensorDataQoS());
    info_pub_ = node_->create_publisher<as2_msgs::msg::PlatformInfo>(
      "platform/info", rclcpp::QoS(10));
    motor_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
      "sensor_measurements/motor_angular_speed", rclcpp::SensorDataQoS());
    course_pub_ = node_->create_publisher<as2_msgs::msg::TrajectorySetpoints>(
      "motion_reference/trajectory", rclcpp::SensorDataQoS());
    thrust_sub_ = node_->create_subscription<as2_msgs::msg::Thrust>(
      "actuator_command/thrust", rclcpp::SensorDataQoS(),
      [this](const as2_msgs::msg::Thrust::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        thrusts_.push_back(*msg);
      });
    status_sub_ = node_->create_subscription<diagnostic_msgs::msg::DiagnosticStatus>(
      "debug/controller/rl_policy/status", rclcpp::QoS(10),
      [this](const diagnostic_msgs::msg::DiagnosticStatus::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        status_.clear();
        for (const auto & entry : msg->values) {
          status_[entry.key] = entry.value;
        }
      });
    set_mode_client_ = node_->create_client<as2_msgs::srv::SetControlMode>(
      "controller/set_control_mode");
    timer_ = node_->create_wall_timer(std::chrono::milliseconds(10), [this]() {publish();});
  }

  std::shared_ptr<rclcpp::Node> node() const {return node_;}

  std::size_t thrustCount()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return thrusts_.size();
  }

  as2_msgs::msg::Thrust lastThrust()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return thrusts_.back();
  }

  std::string status(const std::string & key)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = status_.find(key);
    return it == status_.end() ? std::string() : it->second;
  }

  rclcpp::Client<as2_msgs::srv::SetControlMode>::SharedPtr setModeClient() const
  {
    return set_mode_client_;
  }

private:
  void publish()
  {
    geometry_msgs::msg::TwistStamped twist;
    twist.header.stamp = node_->now();
    twist.header.frame_id = std::string(kNamespace) + "/base_link";
    twist_pub_->publish(twist);

    as2_msgs::msg::PlatformInfo info;
    info.header.stamp = twist.header.stamp;
    info.connected = true;
    info.armed = true;
    info.offboard = true;
    info_pub_->publish(info);

    sensor_msgs::msg::JointState motors;
    motors.header.stamp = twist.header.stamp;
    motors.velocity = {1400.0, 1400.0, 1400.0, 1400.0};
    motor_pub_->publish(motors);

    if (!course_.setpoints.empty()) {
      course_.header.stamp = twist.header.stamp;
      course_pub_->publish(course_);
    }
  }

  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<as2_msgs::msg::PlatformInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr motor_pub_;
  rclcpp::Publisher<as2_msgs::msg::TrajectorySetpoints>::SharedPtr course_pub_;
  as2_msgs::msg::TrajectorySetpoints course_;
  rclcpp::Subscription<as2_msgs::msg::Thrust>::SharedPtr thrust_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr status_sub_;
  rclcpp::Client<as2_msgs::srv::SetControlMode>::SharedPtr set_mode_client_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::mutex mutex_;
  std::vector<as2_msgs::msg::Thrust> thrusts_;
  std::map<std::string, std::string> status_;
};

bool waitFor(const std::function<bool()> & condition, std::chrono::seconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return condition();
}

}  // namespace

TEST(PluginThroughTheManager, FliesOnTrajectoryThenHoldsOnHover) {
  std::string plugins;
  ASSERT_TRUE(
    ament_index_cpp::get_resource(
      "as2_motion_controller__pluginlib__plugin", "as2_rl_policy", plugins))
    << "as2_rl_policy is not in the ament index: build and install the package first";

  auto manager = makeManager();
  rclcpp::NodeOptions platform_options;
  platform_options.arguments({"--ros-args", "-r", std::string("__ns:=/") + kNamespace});
  auto platform = std::make_shared<as2_motion_controller_test::MockPlatform>(
    std::vector<uint8_t>{0b00010000, 0b00100100},
    as2_motion_controller_test::MockPlatform::ControlModeRequest(),
    std::chrono::milliseconds(500), platform_options);
  Drone drone;

  rclcpp::executors::SingleThreadedExecutor manager_executor;
  rclcpp::executors::MultiThreadedExecutor platform_executor;
  rclcpp::executors::SingleThreadedExecutor drone_executor;
  manager_executor.add_node(manager);
  platform_executor.add_node(platform);
  drone_executor.add_node(drone.node());
  std::thread manager_thread([&manager_executor]() {manager_executor.spin();});
  std::thread platform_thread([&platform_executor]() {platform_executor.spin();});
  std::thread drone_thread([&drone_executor]() {drone_executor.spin();});

  // The mock platform asks for TRAJECTORY: the race starts with the first motor speeds.
  const bool flying = waitFor(
    [&drone]() {return drone.thrustCount() >= 10;},
    std::chrono::seconds(20));
  EXPECT_TRUE(flying) << "no command on " << scoped("actuator_command/thrust");
  if (flying) {
    const as2_msgs::msg::Thrust thrust = drone.lastThrust();
    EXPECT_EQ(thrust.header.frame_id, std::string(kNamespace) + "/base_link");
    EXPECT_TRUE(std::isfinite(thrust.thrust));
    EXPECT_TRUE(
      waitFor(
        [&drone]() {return drone.status("phase") == "race";},
        std::chrono::seconds(5)));

    auto request = std::make_shared<as2_msgs::srv::SetControlMode::Request>();
    request->control_mode.control_mode = as2_msgs::msg::ControlMode::HOVER;
    request->control_mode.yaw_mode = as2_msgs::msg::ControlMode::YAW_ANGLE;
    ASSERT_TRUE(drone.setModeClient()->wait_for_service(std::chrono::seconds(5)));
    auto response = drone.setModeClient()->async_send_request(request);
    ASSERT_EQ(response.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_TRUE(response.get()->success);

    EXPECT_TRUE(
      waitFor(
        [&drone]() {return drone.status("phase") == "hold";},
        std::chrono::seconds(10)));
    EXPECT_EQ(drone.status("exit_reason"), "hover_request");
    EXPECT_EQ(drone.status("policy"), "hover");
    EXPECT_NEAR(std::stod(drone.status("setpoint_x")), 14.5, 1e-6);
    EXPECT_NEAR(std::stod(drone.status("setpoint_z")), 1.0, 1e-6);
    const std::size_t before = drone.thrustCount();
    EXPECT_TRUE(
      waitFor(
        [&drone, before]() {return drone.thrustCount() > before + 10;},
        std::chrono::seconds(5)));
  }

  manager_executor.cancel();
  platform_executor.cancel();
  drone_executor.cancel();
  manager_thread.join();
  platform_thread.join();
  drone_thread.join();
}

TEST(PluginThroughTheManager, TheHoverMissionHoldsTheLastPointOfTheSentCourse) {
  std::string plugins;
  ASSERT_TRUE(
    ament_index_cpp::get_resource(
      "as2_motion_controller__pluginlib__plugin", "as2_rl_policy", plugins))
    << "as2_rl_policy is not in the ament index: build and install the package first";

  const std::string policy = rl_policy_test::fixturePath(kPolicyFixture);
  auto manager = makeManager(
    {
      rclcpp::Parameter("rl_policy.mission", std::string("hover")),
      rclcpp::Parameter("rl_policy.policies.names", std::vector<std::string>{"hover"}),
      rclcpp::Parameter("rl_policy.policies.files", std::vector<std::string>{policy}),
    });
  rclcpp::NodeOptions platform_options;
  platform_options.arguments({"--ros-args", "-r", std::string("__ns:=/") + kNamespace});
  auto platform = std::make_shared<as2_motion_controller_test::MockPlatform>(
    std::vector<uint8_t>{0b00010000, 0b00100100},
    as2_motion_controller_test::MockPlatform::ControlModeRequest(),
    std::chrono::milliseconds(500), platform_options);

  // As as2_race_pilot sends it: the vehicle, a gate, then the course's end point.
  as2_msgs::msg::TrajectorySetpoints course;
  course.header.frame_id = "earth";
  for (const auto & values : std::vector<std::vector<double>>{
      {14.5, 2.0, 0.2, 0.0}, {12.5, 2.0, 1.45, 3.14}, {10.0, 5.0, 1.8, 0.5}})
  {
    as2_msgs::msg::TrajectoryPoint point;
    point.position.x = values[0];
    point.position.y = values[1];
    point.position.z = values[2];
    point.yaw_angle = values[3];
    course.setpoints.push_back(point);
  }
  Drone drone(course);

  rclcpp::executors::SingleThreadedExecutor manager_executor;
  rclcpp::executors::MultiThreadedExecutor platform_executor;
  rclcpp::executors::SingleThreadedExecutor drone_executor;
  manager_executor.add_node(manager);
  platform_executor.add_node(platform);
  drone_executor.add_node(drone.node());
  std::thread manager_thread([&manager_executor]() {manager_executor.spin();});
  std::thread platform_thread([&platform_executor]() {platform_executor.spin();});
  std::thread drone_thread([&drone_executor]() {drone_executor.spin();});

  const bool holding = waitFor(
    [&drone]() {
      return drone.status("phase") == "hold" && drone.status("setpoint_source") == "sent";
    },
    std::chrono::seconds(20));
  EXPECT_TRUE(holding) << "no hold at the sent course's end point";
  if (holding) {
    EXPECT_EQ(drone.status("hold_reference"), "sent_or_configured");
    EXPECT_EQ(drone.status("policy"), "hover");
    EXPECT_NEAR(std::stod(drone.status("setpoint_x")), 10.0, 1e-6);
    EXPECT_NEAR(std::stod(drone.status("setpoint_y")), 5.0, 1e-6);
    EXPECT_NEAR(std::stod(drone.status("setpoint_z")), 1.8, 1e-6);
    EXPECT_NEAR(std::stod(drone.status("setpoint_yaw")), 0.5, 1e-6);
    const std::size_t before = drone.thrustCount();
    EXPECT_TRUE(
      waitFor(
        [&drone, before]() {return drone.thrustCount() > before + 10;},
        std::chrono::seconds(5)));
  }

  manager_executor.cancel();
  platform_executor.cancel();
  drone_executor.cancel();
  manager_thread.join();
  platform_thread.join();
  drone_thread.join();
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
