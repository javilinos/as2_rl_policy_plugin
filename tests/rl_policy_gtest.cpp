#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "as2_core/node.hpp"
#include "as2_msgs/msg/control_mode.hpp"
#include "as2_rl_policy/rl_policy.hpp"
#include "fixture_utils.hpp"

namespace
{

using as2_msgs::msg::ControlMode;
using rl_policy::Phase;

constexpr char kNamespace[] = "test_rl_policy";
constexpr char kPolicyFixture[] = "policy_random.yaml";

double policyDt()
{
  return rl_policy_test::loadFixture(kPolicyFixture)["dt"].as<double>();
}

// The season 2 course and a pair of copies of the fixture policy.
std::vector<rclcpp::Parameter> flightParameters(const std::string & policy_file)
{
  const YAML::Node course = rl_policy_test::loadFixture("obs_season2.yaml")["course"];
  return {
    rclcpp::Parameter("use_sim_time", true),
    rclcpp::Parameter("cmd_freq", 1.0 / policyDt()),
    rclcpp::Parameter(
      "rl_policy.policies.files", std::vector<std::string>{policy_file, policy_file}),
    rclcpp::Parameter("rl_policy.course.gates_x", rl_policy_test::doubles(course["gates_x"])),
    rclcpp::Parameter("rl_policy.course.gates_y", rl_policy_test::doubles(course["gates_y"])),
    rclcpp::Parameter("rl_policy.course.gates_z", rl_policy_test::doubles(course["gates_z"])),
    rclcpp::Parameter("rl_policy.course.gates_yaw", rl_policy_test::doubles(course["gates_yaw"])),
  };
}

std::vector<rclcpp::Parameter> with(
  std::vector<rclcpp::Parameter> parameters, const rclcpp::Parameter & parameter)
{
  for (rclcpp::Parameter & existing : parameters) {
    if (existing.get_name() == parameter.get_name()) {
      existing = parameter;
      return parameters;
    }
  }
  parameters.push_back(parameter);
  return parameters;
}

ControlMode controlMode(int8_t control_mode, int8_t yaw_mode)
{
  ControlMode mode;
  mode.control_mode = control_mode;
  mode.yaw_mode = yaw_mode;
  return mode;
}

const ControlMode kTrajectory = controlMode(ControlMode::TRAJECTORY, ControlMode::YAW_ANGLE);
const ControlMode kBodyRates = controlMode(ControlMode::BODY_RATES, ControlMode::YAW_SPEED);

geometry_msgs::msg::PoseStamped pose(double x, double y, double z, double yaw)
{
  geometry_msgs::msg::PoseStamped msg;
  msg.header.frame_id = "earth";
  msg.pose.position.x = x;
  msg.pose.position.y = y;
  msg.pose.position.z = z;
  msg.pose.orientation.w = std::cos(0.5 * yaw);
  msg.pose.orientation.z = std::sin(0.5 * yaw);
  return msg;
}

geometry_msgs::msg::TwistStamped twist()
{
  geometry_msgs::msg::TwistStamped msg;
  msg.header.frame_id = "earth";
  return msg;
}

std::map<std::string, std::string> values(const diagnostic_msgs::msg::DiagnosticStatus & status)
{
  std::map<std::string, std::string> out;
  for (const auto & entry : status.values) {
    out[entry.key] = entry.value;
  }
  return out;
}

class PluginTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    helper_ = std::make_shared<rclcpp::Node>(
      "rl_policy_test_helper", std::string("/") + kNamespace);
    clock_pub_ = helper_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
    motor_pub_ = helper_->create_publisher<sensor_msgs::msg::JointState>(
      "sensor_measurements/motor_speed", rclcpp::SensorDataQoS());
    status_sub_ = helper_->create_subscription<diagnostic_msgs::msg::DiagnosticStatus>(
      std::string("/") + kNamespace + "/debug/controller/rl_policy/status", rclcpp::QoS(10),
      [this](const diagnostic_msgs::msg::DiagnosticStatus::SharedPtr msg) {
        statuses_.push_back(*msg);
      });
    step_sub_ = helper_->create_subscription<std_msgs::msg::Float64MultiArray>(
      std::string("/") + kNamespace + "/debug/controller/rl_policy/step", rclcpp::QoS(10),
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {steps_.push_back(*msg);});
  }

  void TearDown() override
  {
    plugin_.reset();
    node_.reset();
  }

  void build(const std::vector<rclcpp::Parameter> & overrides)
  {
    plugin_.reset();
    node_.reset();
    rclcpp::NodeOptions options;
    options.arguments(
      {"--ros-args", "-r", std::string("__ns:=/") + kNamespace, "--params-file",
        std::string(RL_POLICY_CONFIG_DIR) + "/rl_policy_default.yaml"});
    options.parameter_overrides(overrides);
    options.automatically_declare_parameters_from_overrides(true);
    node_ = std::make_shared<as2::Node>("controller_manager", options);

    // What ControllerManager does before and after initialize().
    plugin_ = std::make_shared<rl_policy::Plugin>();
    plugin_->setPluginParamNamespace("rl_policy");
    plugin_->initialize(node_.get());
    plugin_->reset();
    std::vector<rclcpp::Parameter> all;
    for (const auto & name : node_->list_parameters({}, 0).names) {
      all.push_back(node_->get_parameter(name));
    }
    plugin_->dispatchParameters(all);
  }

  void spin(std::chrono::milliseconds duration = std::chrono::milliseconds(20))
  {
    const auto deadline = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < deadline) {
      rclcpp::spin_some(node_);
      rclcpp::spin_some(helper_);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Publishes the sim time until the node's clock shows it.
  void setTime(double seconds)
  {
    rosgraph_msgs::msg::Clock clock;
    clock.clock = rclcpp::Time(static_cast<int64_t>(std::llround(seconds * 1e9)), RCL_ROS_TIME);
    const rclcpp::Time target(clock.clock, RCL_ROS_TIME);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      clock_pub_->publish(clock);
      rclcpp::spin_some(node_);
      if (node_->now() == target) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    FAIL() << "the node clock never reached " << seconds;
  }

  void publishMotorSpeeds(double speed)
  {
    sensor_msgs::msg::JointState msg;
    msg.velocity = {speed, speed + 10.0, speed + 20.0, speed + 30.0};
    motor_pub_->publish(msg);
  }

  // Sets the TRAJECTORY mode as the handler does, then feeds one state.
  void setTrajectoryMode(const geometry_msgs::msg::PoseStamped & state_pose)
  {
    ASSERT_TRUE(plugin_->setMode(kTrajectory, kBodyRates));
    plugin_->reset();
    plugin_->setHoverEnabled(false);
    plugin_->updateState(state_pose, twist());
  }

  // Calls computeOutput until the motor speeds are in and a step is taken.
  bool firstStep()
  {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      publishMotorSpeeds(1500.0);
      spin(std::chrono::milliseconds(5));
      if (plugin_->computeOutput(0.0, pose_out_, twist_out_, thrust_out_)) {
        return true;
      }
    }
    return false;
  }

  bool computeOutput()
  {
    return plugin_->computeOutput(0.0, pose_out_, twist_out_, thrust_out_);
  }

  // The status message the timer publishes once the node clock moves: the refusals, if any.
  std::string statusMessage()
  {
    statuses_.clear();
    setTime(1.0);
    spin(std::chrono::milliseconds(300));
    return statuses_.empty() ? std::string() : statuses_.back().message;
  }

  std::shared_ptr<rclcpp::Node> helper_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr motor_pub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr step_sub_;
  std::vector<diagnostic_msgs::msg::DiagnosticStatus> statuses_;
  std::vector<std_msgs::msg::Float64MultiArray> steps_;

  std::shared_ptr<as2::Node> node_;
  std::shared_ptr<rl_policy::Plugin> plugin_;
  geometry_msgs::msg::PoseStamped pose_out_;
  geometry_msgs::msg::TwistStamped twist_out_;
  as2_msgs::msg::Thrust thrust_out_;
};

}  // namespace

TEST_F(PluginTest, LoadsAndAcceptsItsModePairOnly) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  ASSERT_NE(plugin_->mission(), nullptr);
  EXPECT_EQ(plugin_->mission()->phase(), Phase::Idle);
  EXPECT_EQ(plugin_->getDesiredPoseFrameId(), "earth");
  EXPECT_EQ(plugin_->getDesiredTwistFrameId(), "earth");
  EXPECT_EQ(plugin_->mission()->config().gates.size(), 12u);

  const ControlMode hover = plugin_->hoverMode();
  EXPECT_EQ(hover.control_mode, ControlMode::TRAJECTORY);
  EXPECT_EQ(hover.yaw_mode, ControlMode::YAW_ANGLE);
  EXPECT_TRUE(plugin_->setMode(kTrajectory, kBodyRates));
  const ControlMode position = controlMode(ControlMode::POSITION, ControlMode::YAW_ANGLE);
  const ControlMode attitude = controlMode(ControlMode::ATTITUDE, ControlMode::YAW_ANGLE);
  const ControlMode rates_yaw_angle = controlMode(ControlMode::BODY_RATES, ControlMode::YAW_ANGLE);
  EXPECT_FALSE(plugin_->setMode(position, kBodyRates));
  EXPECT_FALSE(plugin_->setMode(kTrajectory, attitude));
  EXPECT_FALSE(plugin_->setMode(kTrajectory, rates_yaw_angle));
}

TEST_F(PluginTest, RefusesACorruptedPolicyFile) {
  YAML::Node node = YAML::Clone(rl_policy_test::loadFixture(kPolicyFixture));
  YAML::Node weight = node["network"]["layers"][3]["weight"];
  weight[70] = weight[70].as<double>() - 1.0;
  build(flightParameters(rl_policy_test::writeYaml(node, "corrupted")));
  EXPECT_FALSE(plugin_->isReady());
  EXPECT_EQ(plugin_->mission(), nullptr);
  EXPECT_FALSE(plugin_->setMode(kTrajectory, kBodyRates));
}

TEST_F(PluginTest, RefusesACmdFreqThatIsNotTheStep) {
  const auto parameters = flightParameters(rl_policy_test::fixturePath(kPolicyFixture));
  build(with(parameters, rclcpp::Parameter("cmd_freq", 0.5 / policyDt())));
  EXPECT_FALSE(plugin_->isReady());
  EXPECT_FALSE(plugin_->setMode(kTrajectory, kBodyRates));
}

TEST_F(PluginTest, RefusesFramesOtherThanEarth) {
  const auto parameters = flightParameters(rl_policy_test::fixturePath(kPolicyFixture));
  build(with(parameters, rclcpp::Parameter("desired_pose_frame", std::string("odom"))));
  EXPECT_FALSE(plugin_->isReady());
  EXPECT_FALSE(plugin_->setMode(kTrajectory, kBodyRates));
  build(with(parameters, rclcpp::Parameter("desired_twist_frame", std::string(""))));
  EXPECT_FALSE(plugin_->isReady());
}

TEST_F(PluginTest, RefusesParametersOfTheWrongType) {
  const auto parameters = flightParameters(rl_policy_test::fixturePath(kPolicyFixture));
  build(with(parameters, rclcpp::Parameter("rl_policy.race.laps", 3.0)));
  EXPECT_FALSE(plugin_->isReady());
  build(with(parameters, rclcpp::Parameter("rl_policy.vehicle.mass_kg", 1)));
  EXPECT_FALSE(plugin_->isReady());
  build(with(parameters, rclcpp::Parameter("rl_policy.config_version", 2)));
  EXPECT_FALSE(plugin_->isReady());
  build(with(parameters, rclcpp::Parameter("rl_policy.mission", std::string("land"))));
  EXPECT_FALSE(plugin_->isReady());
  build(
    with(
      parameters, rclcpp::Parameter("rl_policy.course.gates_z", std::vector<double>{1.0, 2.0})));
  EXPECT_FALSE(plugin_->isReady());
  build(
    with(
      parameters, rclcpp::Parameter("rl_policy.policies.names", std::vector<std::string>{"race"})));
  EXPECT_FALSE(plugin_->isReady());
  build(
    with(
      parameters,
      rclcpp::Parameter("rl_policy.policies.names", std::vector<std::string>{"race", "other"})));
  EXPECT_FALSE(plugin_->isReady());
}

TEST_F(PluginTest, HoverMissionNeedsOnlyTheHoverPolicy) {
  auto parameters = flightParameters(rl_policy_test::fixturePath(kPolicyFixture));
  parameters = with(parameters, rclcpp::Parameter("rl_policy.mission", std::string("hover")));
  parameters = with(
    parameters, rclcpp::Parameter("rl_policy.policies.names", std::vector<std::string>{"hover"}));
  parameters = with(
    parameters, rclcpp::Parameter(
      "rl_policy.policies.files",
      std::vector<std::string>{rl_policy_test::fixturePath(kPolicyFixture)}));
  build(parameters);
  ASSERT_TRUE(plugin_->isReady());
  EXPECT_EQ(plugin_->mission()->config().type, rl_policy::MissionType::Hover);
}

TEST_F(PluginTest, AfterRaceIsHereOrASetpointInsideTheBounds) {
  auto parameters = flightParameters(rl_policy_test::fixturePath(kPolicyFixture));
  parameters = with(
    parameters, rclcpp::Parameter("rl_policy.course.bounds_x", std::vector<double>{1.0, 22.0}));
  parameters = with(
    parameters, rclcpp::Parameter("rl_policy.course.bounds_y", std::vector<double>{1.0, 32.0}));
  const auto after_race = [](const std::string & value) {
      return rclcpp::Parameter("rl_policy.hold.after_race", value);
    };
  const auto hold_at = [](double x, double y) {
      return rclcpp::Parameter("rl_policy.hold.setpoint", std::vector<double>{x, y, 0.2, 3.14});
    };
  const auto inside = with(parameters, hold_at(14.5, 2.0));

  build(with(inside, after_race("start")));
  EXPECT_FALSE(plugin_->isReady());
  EXPECT_FALSE(plugin_->setMode(kTrajectory, kBodyRates));
  EXPECT_NE(statusMessage().find("'start' is neither 'here' nor 'setpoint'"), std::string::npos);

  build(with(inside, rclcpp::Parameter("rl_policy.hold.after_race", 1)));
  EXPECT_FALSE(plugin_->isReady());
  EXPECT_FALSE(plugin_->setMode(kTrajectory, kBodyRates));
  EXPECT_NE(statusMessage().find("hold.after_race must be of type string"), std::string::npos);

  build(
    with(
      inside,
      rclcpp::Parameter("rl_policy.hold.after_race", std::vector<std::string>{"setpoint"})));
  EXPECT_FALSE(plugin_->isReady());

  build(with(inside, after_race("setpoint")));
  ASSERT_TRUE(plugin_->isReady());
  EXPECT_EQ(plugin_->mission()->config().after_race, rl_policy::AfterRace::Setpoint);
  EXPECT_EQ(
    plugin_->mission()->config().hold_setpoint.position, Eigen::Vector3d(14.5, 2.0, 0.2));
  EXPECT_TRUE(plugin_->setMode(kTrajectory, kBodyRates));

  for (const auto & outside : {hold_at(22.5, 2.0), hold_at(14.5, 0.5), hold_at(0.0, 0.0)}) {
    build(with(with(parameters, outside), after_race("setpoint")));
    EXPECT_FALSE(plugin_->isReady()) << outside.value_to_string();
    EXPECT_FALSE(plugin_->setMode(kTrajectory, kBodyRates));
    EXPECT_NE(
      statusMessage().find("hold.setpoint must lie inside the course bounds"), std::string::npos)
      << outside.value_to_string();
  }

  // With here the race never holds at the setpoint, so the bounds do not constrain it.
  build(with(with(parameters, hold_at(22.5, 2.0)), after_race("here")));
  ASSERT_TRUE(plugin_->isReady());
  EXPECT_EQ(plugin_->mission()->config().after_race, rl_policy::AfterRace::Here);
}

TEST_F(PluginTest, ReferenceIsMarkedAfterResetAndAState) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  ASSERT_TRUE(plugin_->setMode(kTrajectory, kBodyRates));
  plugin_->reset();
  EXPECT_FALSE(plugin_->isReferenceReceived());
  auto wrong_frame = pose(14.5, 2.0, 0.2, 0.0);
  wrong_frame.header.frame_id = std::string(kNamespace) + "/odom";
  plugin_->updateState(wrong_frame, twist());
  EXPECT_FALSE(plugin_->isReferenceReceived());
  plugin_->updateState(pose(14.5, 2.0, 0.2, 0.0), twist());
  EXPECT_TRUE(plugin_->isReferenceReceived());
}

TEST_F(PluginTest, HoverRequestHoldsAtTheClampedPoseOnce) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  const rl_policy::MissionConfig & config = plugin_->mission()->config();

  // The handler: reset(), then the hover flag, then the frozen pose reference.
  ASSERT_TRUE(plugin_->setMode(kTrajectory, kBodyRates));
  plugin_->reset();
  plugin_->setHoverEnabled(true);
  plugin_->updateReference(pose(0.5, 40.0, 0.2, 0.3));
  ASSERT_EQ(plugin_->mission()->phase(), Phase::Hold);
  const rl_policy::MissionStatus & status = plugin_->mission()->status();
  EXPECT_EQ(status.exit_reason, rl_policy::ExitReason::HoverRequest);
  EXPECT_EQ(status.policy, "hover");
  EXPECT_DOUBLE_EQ(status.setpoint.position.x(), config.bounds.x_min);
  EXPECT_DOUBLE_EQ(status.setpoint.position.y(), config.bounds.y_max);
  EXPECT_DOUBLE_EQ(status.setpoint.position.z(), config.min_altitude_m);
  EXPECT_NEAR(status.setpoint.yaw, 0.3, 1e-12);

  plugin_->updateReference(twist());
  plugin_->updateReference(pose(5.0, 5.0, 5.0, 0.0));
  EXPECT_DOUBLE_EQ(plugin_->mission()->status().setpoint.position.x(), config.bounds.x_min);

  // A TRAJECTORY mode set does not take pose references.
  plugin_->reset();
  plugin_->setHoverEnabled(false);
  plugin_->updateReference(pose(6.0, 6.0, 6.0, 0.0));
  EXPECT_DOUBLE_EQ(plugin_->mission()->status().setpoint.position.x(), config.bounds.x_min);

  // A new HOVER mode set takes the next one.
  plugin_->reset();
  plugin_->setHoverEnabled(true);
  plugin_->updateReference(pose(5.0, 6.0, 7.0, -0.2));
  EXPECT_EQ(plugin_->mission()->status().setpoint.position, Eigen::Vector3d(5.0, 6.0, 7.0));
  EXPECT_EQ(plugin_->mission()->phase(), Phase::Hold);
}

TEST_F(PluginTest, HoverModeFromIdleWaitsForItsSetpoint) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  setTime(30.0);
  ASSERT_TRUE(plugin_->setMode(kTrajectory, kBodyRates));
  plugin_->reset();
  plugin_->setHoverEnabled(true);
  plugin_->updateReference(pose(std::nan(""), 2.0, 0.2, 3.14));
  plugin_->updateState(pose(14.5, 2.0, 0.2, 3.14), twist());
  for (int i = 0; i < 20; ++i) {
    publishMotorSpeeds(1500.0);
    spin(std::chrono::milliseconds(5));
  }
  // No race in a HOVER mode, whatever the phase.
  EXPECT_FALSE(computeOutput());
  EXPECT_EQ(plugin_->mission()->phase(), Phase::Idle);

  plugin_->updateReference(pose(14.5, 2.0, 0.2, 3.14));
  ASSERT_EQ(plugin_->mission()->phase(), Phase::Hold);
  ASSERT_TRUE(firstStep());
  EXPECT_EQ(plugin_->mission()->status().policy, "hover");
  EXPECT_EQ(plugin_->mission()->status().steps, 1u);
}

TEST_F(PluginTest, StepsOncePerPeriodAndHoldsTheCommandInBetween) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  const double dt = policyDt();
  setTime(100.0);
  setTrajectoryMode(pose(14.5, 2.0, 0.2, 3.14));

  // Nothing before the motor speeds: the drone is still on the floor.
  EXPECT_FALSE(computeOutput());
  EXPECT_EQ(plugin_->mission()->phase(), Phase::Idle);
  ASSERT_TRUE(firstStep());
  EXPECT_EQ(plugin_->mission()->phase(), Phase::Race);
  EXPECT_EQ(plugin_->mission()->status().steps, 1u);

  const std::string base_link = std::string(kNamespace) + "/base_link";
  EXPECT_EQ(thrust_out_.header.frame_id, base_link);
  EXPECT_EQ(twist_out_.header.frame_id, base_link);
  EXPECT_EQ(rclcpp::Time(thrust_out_.header.stamp, RCL_ROS_TIME), node_->now());
  EXPECT_TRUE(std::isfinite(thrust_out_.thrust));
  EXPECT_GE(thrust_out_.thrust, 0.0);
  EXPECT_TRUE(std::isfinite(twist_out_.twist.angular.x));
  EXPECT_TRUE(std::isfinite(twist_out_.twist.angular.y));
  EXPECT_TRUE(std::isfinite(twist_out_.twist.angular.z));
  const auto thrust = thrust_out_.thrust;
  const auto rates = twist_out_.twist.angular;

  // An early call republishes the held command.
  thrust_out_ = as2_msgs::msg::Thrust();
  twist_out_ = geometry_msgs::msg::TwistStamped();
  setTime(100.0 + 0.4 * dt);
  ASSERT_TRUE(computeOutput());
  EXPECT_EQ(plugin_->mission()->status().steps, 1u);
  EXPECT_EQ(thrust_out_.thrust, thrust);
  EXPECT_EQ(twist_out_.twist.angular, rates);
  EXPECT_EQ(thrust_out_.header.frame_id, base_link);

  setTime(100.0 + dt);
  ASSERT_TRUE(computeOutput());
  EXPECT_EQ(plugin_->mission()->status().steps, 2u);
  EXPECT_EQ(plugin_->overruns(), 0u);

  // A late one steps and is counted.
  setTime(100.0 + 3.0 * dt);
  ASSERT_TRUE(computeOutput());
  EXPECT_EQ(plugin_->mission()->status().steps, 3u);
  EXPECT_EQ(plugin_->overruns(), 1u);

  // The debug topics, under the controller's debug namespace.
  spin(std::chrono::milliseconds(300));
  ASSERT_EQ(steps_.size(), 3u);
  const std_msgs::msg::Float64MultiArray & step = steps_.back();
  ASSERT_EQ(step.layout.dim.size(), 1u);
  EXPECT_EQ(step.layout.dim[0].label, rl_policy::kStepLabel);
  EXPECT_EQ(step.layout.dim[0].size, 61u);
  ASSERT_EQ(step.data.size(), 61u);
  EXPECT_NEAR(step.data[0], 100.0 + 3.0 * dt, 1e-9);
  EXPECT_EQ(step.data[1], 1.0);
  EXPECT_EQ(step.data[2], 0.0);
  EXPECT_DOUBLE_EQ(step.data[8], 14.5);
  EXPECT_DOUBLE_EQ(step.data[21], 1500.0);
  EXPECT_DOUBLE_EQ(step.data[24], 1530.0);
  EXPECT_NEAR(step.data[25 + 12], 2.0 * 1500.0 / 3000.0 - 1.0, 1e-6);
  EXPECT_NEAR(step.data[57], thrust_out_.thrust, 1e-4);
  EXPECT_NEAR(step.data[58], twist_out_.twist.angular.x, 1e-12);

  // The status timer runs on the node clock.
  setTime(100.25);
  spin(std::chrono::milliseconds(300));
  ASSERT_FALSE(statuses_.empty());
  EXPECT_EQ(values(statuses_.back())["phase"], "race");
  EXPECT_EQ(values(statuses_.back())["overruns"], "1");
  EXPECT_EQ(values(statuses_.back())["steps"], "3");
}

TEST_F(PluginTest, StaleMotorSpeedsAreFlaggedNotRefused) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  setTime(50.0);
  setTrajectoryMode(pose(14.5, 2.0, 0.2, 3.14));
  ASSERT_TRUE(firstStep());
  const double timeout = node_->get_parameter("rl_policy.motor_speed.timeout_s").as_double();
  setTime(50.0 + 3.0 * timeout);
  EXPECT_TRUE(computeOutput());
  spin(std::chrono::milliseconds(300));
  ASSERT_FALSE(statuses_.empty());
  bool flagged = false;
  for (const auto & status : statuses_) {
    flagged = flagged || (values(status)["motor_speed_stale"] == "true" &&
      status.level == diagnostic_msgs::msg::DiagnosticStatus::WARN);
  }
  EXPECT_TRUE(flagged);
}

TEST_F(PluginTest, NonFiniteStateGivesNoOutput) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  setTime(10.0);
  setTrajectoryMode(pose(14.5, 2.0, 0.2, 3.14));
  ASSERT_TRUE(firstStep());
  plugin_->updateState(pose(std::nan(""), 2.0, 0.2, 3.14), twist());
  setTime(10.0 + policyDt());
  EXPECT_FALSE(computeOutput());
}

TEST_F(PluginTest, TrajectoryModeSetAfterAHoldIsANoOp) {
  build(flightParameters(rl_policy_test::fixturePath(kPolicyFixture)));
  ASSERT_TRUE(plugin_->isReady());
  setTime(20.0);
  setTrajectoryMode(pose(14.5, 2.0, 0.2, 3.14));
  ASSERT_TRUE(firstStep());
  plugin_->reset();
  plugin_->setHoverEnabled(true);
  plugin_->updateReference(pose(14.5, 2.0, 1.5, 3.14));
  ASSERT_EQ(plugin_->mission()->phase(), Phase::Hold);

  setTrajectoryMode(pose(14.5, 2.0, 1.5, 3.14));
  setTime(20.0 + policyDt());
  ASSERT_TRUE(computeOutput());
  EXPECT_EQ(plugin_->mission()->phase(), Phase::Hold);
  EXPECT_EQ(plugin_->mission()->status().policy, "hover");
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
