#include "as2_rl_policy/rl_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

#include "as2_core/utils/control_mode_utils.hpp"
#include "as2_rl_policy/core/angles.hpp"

namespace rl_policy
{

namespace
{

constexpr int64_t kConfigVersion = 1;
// cmd_freq times the policy dt must be one to this tolerance.
constexpr double kStepRateTolerance = 1.0e-6;

std::string text(double value)
{
  return std::to_string(value);
}

std::string text(bool value)
{
  return value ? "true" : "false";
}

diagnostic_msgs::msg::KeyValue keyValue(const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue entry;
  entry.key = key;
  entry.value = value;
  return entry;
}

as2_msgs::msg::ControlMode controlMode(int8_t control_mode, int8_t yaw_mode)
{
  as2_msgs::msg::ControlMode mode;
  mode.control_mode = control_mode;
  mode.yaw_mode = yaw_mode;
  return mode;
}

}  // namespace

std::vector<std::string> Plugin::initParameters() const
{
  return {
    "config_version",
    "mission",
    "policies.names",
    "policies.files",
    "vehicle.mass_kg",
    "motor_speed.topic",
    "motor_speed.timeout_s",
    "course.gates_x",
    "course.gates_y",
    "course.gates_z",
    "course.gates_yaw",
    "course.bounds_x",
    "course.bounds_y",
    "race.laps",
    "race.pass_tolerance_m",
    "race.valid_half_m",
    "race.gate_timeout_s",
    "race.bounds_margin_m",
    "race.ceiling_m",
    "race.exit_debounce_steps",
    "hold.setpoint",
    "hold.min_altitude_m",
  };
}

void Plugin::updateParameter(const std::string & name, const rclcpp::Parameter & parameter)
{
  const std::vector<std::string> tails = initParameters();
  if (std::find(tails.begin(), tails.end(), name) == tails.end()) {
    RCLCPP_WARN(
      getNodePtr()->get_logger(), "Unknown parameter '%s' is ignored", param(name).c_str());
    return;
  }
  parameters_.insert_or_assign(name, parameter);
}

void Plugin::refuse(const std::string & reason)
{
  refusals_.push_back(reason);
}

const rclcpp::Parameter * Plugin::setting(const std::string & tail, rclcpp::ParameterType type)
{
  const auto it = parameters_.find(tail);
  if (it == parameters_.end()) {
    refuse(param(tail) + " is not set by any parameter file");
    return nullptr;
  }
  const rclcpp::ParameterType actual = it->second.get_type();
  if (actual != type) {
    using rclcpp::ParameterType;
    const bool integer = type == ParameterType::PARAMETER_DOUBLE &&
      actual == ParameterType::PARAMETER_INTEGER;
    const bool integers = type == ParameterType::PARAMETER_DOUBLE_ARRAY &&
      actual == ParameterType::PARAMETER_INTEGER_ARRAY;
    std::string hint;
    if (integer) {
      hint = " (write 1.0, not 1)";
    } else if (integers) {
      hint = " (write every value with a decimal point)";
    }
    refuse(
      param(tail) + " must be of type " + rclcpp::to_string(type) + ", it is " +
      rclcpp::to_string(actual) + hint);
    return nullptr;
  }
  return &it->second;
}

void Plugin::readSettings(Settings & settings)
{
  using rclcpp::ParameterType;
  MissionConfig & mission = settings.mission;

  if (const auto * p = setting("config_version", ParameterType::PARAMETER_INTEGER)) {
    if (p->as_int() != kConfigVersion) {
      refuse(
        param("config_version") + " is " + std::to_string(p->as_int()) +
        ", this plugin reads version " + std::to_string(kConfigVersion));
    }
  }
  if (const auto * p = setting("mission", ParameterType::PARAMETER_STRING)) {
    try {
      mission.type = missionTypeFromString(p->as_string());
    } catch (const std::invalid_argument & e) {
      refuse(param("mission") + ": " + e.what());
    }
  }
  const auto * names = setting("policies.names", ParameterType::PARAMETER_STRING_ARRAY);
  const auto * files = setting("policies.files", ParameterType::PARAMETER_STRING_ARRAY);
  if (names && files) {
    settings.policy_names = names->as_string_array();
    settings.policy_files = files->as_string_array();
    settings.policies_named = true;
  }
  if (const auto * p = setting("vehicle.mass_kg", ParameterType::PARAMETER_DOUBLE)) {
    mission.mass_kg = p->as_double();
  }
  if (const auto * p = setting("motor_speed.topic", ParameterType::PARAMETER_STRING)) {
    settings.motor_speed_topic = p->as_string();
    if (settings.motor_speed_topic.empty()) {
      refuse(param("motor_speed.topic") + " is empty: the observation needs the rotor speeds");
    }
  }
  if (const auto * p = setting("motor_speed.timeout_s", ParameterType::PARAMETER_DOUBLE)) {
    settings.motor_speed_timeout_s = p->as_double();
    if (!(settings.motor_speed_timeout_s > 0.0)) {
      refuse(param("motor_speed.timeout_s") + " must be positive");
    }
  }

  const std::vector<std::string> gate_tails = {
    "course.gates_x", "course.gates_y", "course.gates_z", "course.gates_yaw"};
  std::vector<std::vector<double>> gate_columns;
  for (const std::string & tail : gate_tails) {
    if (const auto * p = setting(tail, ParameterType::PARAMETER_DOUBLE_ARRAY)) {
      gate_columns.push_back(p->as_double_array());
    }
  }
  if (gate_columns.size() == gate_tails.size()) {
    const std::size_t n = gate_columns[0].size();
    bool same_length = n >= 1;
    for (const auto & column : gate_columns) {
      same_length = same_length && column.size() == n;
    }
    if (!same_length) {
      refuse(
        param("course.gates_*") + " must be four arrays of the same length, at least one gate");
    } else {
      mission.gates.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        mission.gates[i].position =
          Eigen::Vector3d(gate_columns[0][i], gate_columns[1][i], gate_columns[2][i]);
        mission.gates[i].yaw = gate_columns[3][i];
      }
    }
  }
  if (const auto * p = setting("course.bounds_x", ParameterType::PARAMETER_DOUBLE_ARRAY)) {
    const auto values = p->as_double_array();
    if (values.size() != 2) {
      refuse(param("course.bounds_x") + " must be [min, max]");
    } else {
      mission.bounds.x_min = values[0];
      mission.bounds.x_max = values[1];
    }
  }
  if (const auto * p = setting("course.bounds_y", ParameterType::PARAMETER_DOUBLE_ARRAY)) {
    const auto values = p->as_double_array();
    if (values.size() != 2) {
      refuse(param("course.bounds_y") + " must be [min, max]");
    } else {
      mission.bounds.y_min = values[0];
      mission.bounds.y_max = values[1];
    }
  }

  if (const auto * p = setting("race.laps", ParameterType::PARAMETER_INTEGER)) {
    mission.laps = static_cast<int>(std::clamp<int64_t>(
        p->as_int(), 0, std::numeric_limits<int>::max()));
  }
  if (const auto * p = setting("race.pass_tolerance_m", ParameterType::PARAMETER_DOUBLE)) {
    mission.pass_tolerance_m = p->as_double();
  }
  if (const auto * p = setting("race.valid_half_m", ParameterType::PARAMETER_DOUBLE)) {
    mission.valid_half_m = p->as_double();
  }
  if (const auto * p = setting("race.gate_timeout_s", ParameterType::PARAMETER_DOUBLE)) {
    mission.gate_timeout_s = p->as_double();
  }
  if (const auto * p = setting("race.bounds_margin_m", ParameterType::PARAMETER_DOUBLE)) {
    mission.bounds_margin_m = p->as_double();
  }
  if (const auto * p = setting("race.ceiling_m", ParameterType::PARAMETER_DOUBLE)) {
    mission.ceiling_m = p->as_double();
  }
  if (const auto * p = setting("race.exit_debounce_steps", ParameterType::PARAMETER_INTEGER)) {
    mission.exit_debounce_steps = static_cast<int>(std::clamp<int64_t>(
        p->as_int(), 0, std::numeric_limits<int>::max()));
  }
  if (const auto * p = setting("hold.setpoint", ParameterType::PARAMETER_DOUBLE_ARRAY)) {
    const auto values = p->as_double_array();
    if (values.size() != 4) {
      refuse(param("hold.setpoint") + " must be [x, y, z, yaw]");
    } else {
      mission.hold_setpoint.position = Eigen::Vector3d(values[0], values[1], values[2]);
      mission.hold_setpoint.yaw = values[3];
    }
  }
  if (const auto * p = setting("hold.min_altitude_m", ParameterType::PARAMETER_DOUBLE)) {
    mission.min_altitude_m = p->as_double();
  }
}

void Plugin::ownInitialize()
{
  const rclcpp::Logger logger = getNodePtr()->get_logger();
  status_pub_ = createDebugPublisher<diagnostic_msgs::msg::DiagnosticStatus>(
    "debug.status_topic", rclcpp::QoS(10));
  step_pub_ = createDebugPublisher<std_msgs::msg::Float64MultiArray>(
    "debug.step_topic", rclcpp::QoS(10));

  Settings settings;
  readSettings(settings);

  const std::string & earth = getNodePtr()->getEarthFrameId();
  if (getDesiredPoseFrameId() != earth) {
    refuse(
      "desired_pose_frame resolves to '" + getDesiredPoseFrameId() + "', the course is in '" +
      earth + "': set it to /earth");
  }
  if (getDesiredTwistFrameId() != earth) {
    refuse(
      "desired_twist_frame resolves to '" + getDesiredTwistFrameId() +
      "', the observation reads the world velocity in '" + earth + "': set it to /earth");
  }

  const auto node_double = [this](const std::string & name) {
      if (!getNodePtr()->has_parameter(name)) {
        refuse("node parameter " + name + " is not set");
        return 0.0;
      }
      const rclcpp::Parameter parameter = getNodePtr()->get_parameter(name);
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
        !(parameter.as_double() > 0.0))
      {
        refuse("node parameter " + name + " must be a positive double");
        return 0.0;
      }
      return parameter.as_double();
    };
  const double cmd_freq = node_double("cmd_freq");
  const double info_freq = node_double("info_freq");

  // The policies load whenever they are named, so one launch reports every problem.
  std::unique_ptr<PolicyBank> bank;
  if (settings.policies_named) {
    try {
      bank = std::make_unique<PolicyBank>(
        PolicyBank::load(settings.policy_names, settings.policy_files));
      dt_ = bank->dt();
      for (const std::string & name : bank->names()) {
        const auto policy = bank->get(name);
        const PolicySpec & spec = policy->spec();
        RCLCPP_INFO(
          logger, "Policy '%s' from %s: task %s, dt %.4f s, %d observations", name.c_str(),
          spec.file.c_str(), spec.task.c_str(), spec.dt, spec.observation.dim);
        RCLCPP_INFO(
          logger, "Policy '%s' checkpoint %s sha256 %s", name.c_str(),
          spec.source.checkpoint.c_str(), spec.source.checkpoint_sha256.c_str());
        if (const auto mlp = std::dynamic_pointer_cast<const MlpPolicy>(policy)) {
          const MlpPolicy::GoldenCheck & check = mlp->goldenCheck();
          RCLCPP_INFO(
            logger, "Policy '%s' reproduces its %zu golden rows: max error %.3g <= %.3g",
            name.c_str(), check.rows, check.max_error, check.tolerance);
        }
        if ((name == kRacePolicy || name == kHoverPolicy) && spec.task != name) {
          RCLCPP_WARN(
            logger, "Policy slot '%s' holds a policy trained for task '%s'", name.c_str(),
            spec.task.c_str());
        }
        if (spec.observation.dim > static_cast<int>(kStepObservationColumns)) {
          RCLCPP_WARN(
            logger, "The step topic carries the first %zu of the %d observations of '%s'",
            kStepObservationColumns, spec.observation.dim, name.c_str());
        }
      }
      if (cmd_freq > 0.0 && std::abs(cmd_freq * dt_ - 1.0) > kStepRateTolerance) {
        refuse(
          "cmd_freq is " + text(cmd_freq) + " Hz, the policies step at dt " + text(dt_) +
          " s: set cmd_freq to " + text(1.0 / dt_));
      }
    } catch (const std::exception & e) {
      bank.reset();
      refuse(e.what());
    }
  }
  if (refusals_.empty() && bank) {
    try {
      mission_ = std::make_unique<MissionController>(settings.mission, std::move(*bank));
    } catch (const std::exception & e) {
      refuse(e.what());
    }
  }

  if (!refusals_.empty()) {
    mission_.reset();
    ready_ = false;
    for (const std::string & reason : refusals_) {
      RCLCPP_FATAL(logger, "rl_policy: %s", reason.c_str());
    }
    RCLCPP_FATAL(logger, "rl_policy is not ready: every control mode will be refused");
  } else {
    motor_speed_topic_ = settings.motor_speed_topic;
    motor_speed_timeout_s_ = settings.motor_speed_timeout_s;
    motor_speed_sub_ = getNodePtr()->create_subscription<sensor_msgs::msg::JointState>(
      motor_speed_topic_, rclcpp::SensorDataQoS(),
      std::bind(&Plugin::motorSpeedCallback, this, std::placeholders::_1));
    ready_ = true;
    const MissionConfig & config = mission_->config();
    RCLCPP_INFO(
      logger,
      "rl_policy ready: %s mission, %zu gates, %d laps, step %.4f s, motor speeds from '%s'",
      toString(config.type), config.gates.size(), config.laps, dt_,
      motor_speed_sub_->get_topic_name());
  }

  if (info_freq > 0.0) {
    status_timer_ = getNodePtr()->create_timer(
      std::chrono::duration<double>(1.0 / info_freq), [this]() {publishStatus();});
  }
  publishStatus();
}

void Plugin::reset()
{
  ControllerBase::reset();
  hover_reference_armed_ = true;
  mode_set_pending_ = true;
}

as2_msgs::msg::ControlMode Plugin::hoverMode() const
{
  return controlMode(
    as2_msgs::msg::ControlMode::TRAJECTORY, as2_msgs::msg::ControlMode::YAW_ANGLE);
}

bool Plugin::onSetMode(
  const as2_msgs::msg::ControlMode & mode_in,
  const as2_msgs::msg::ControlMode & mode_out)
{
  if (!ready_) {
    RCLCPP_ERROR(
      getNodePtr()->get_logger(), "rl_policy is not ready, control mode refused: %s",
      refusals_.empty() ? "not initialized" : refusals_.front().c_str());
    return false;
  }
  const bool input_ok = mode_in.control_mode == as2_msgs::msg::ControlMode::TRAJECTORY &&
    mode_in.yaw_mode == as2_msgs::msg::ControlMode::YAW_ANGLE;
  const bool output_ok = mode_out.control_mode == as2_msgs::msg::ControlMode::BODY_RATES &&
    mode_out.yaw_mode == as2_msgs::msg::ControlMode::YAW_SPEED;
  if (!input_ok || !output_ok) {
    RCLCPP_ERROR(
      getNodePtr()->get_logger(),
      "rl_policy takes TRAJECTORY YAW_ANGLE and gives BODY_RATES YAW_SPEED, refused [%s] -> [%s]",
      as2::control_mode::controlModeToString(mode_in).c_str(),
      as2::control_mode::controlModeToString(mode_out).c_str());
    return false;
  }
  return true;
}

void Plugin::onUpdateState(
  const geometry_msgs::msg::PoseStamped & pose_msg,
  const geometry_msgs::msg::TwistStamped & twist_msg)
{
  const auto & position = pose_msg.pose.position;
  const auto & orientation = pose_msg.pose.orientation;
  const auto & twist = twist_msg.twist;
  state_.position = Eigen::Vector3d(position.x, position.y, position.z);
  state_.orientation =
    Eigen::Quaterniond(orientation.w, orientation.x, orientation.y, orientation.z);
  state_.velocity = Eigen::Vector3d(twist.linear.x, twist.linear.y, twist.linear.z);
  state_.body_rates = Eigen::Vector3d(twist.angular.x, twist.angular.y, twist.angular.z);
  state_valid_ = true;
  if (!ready_) {
    return;
  }

  if (mode_set_pending_) {
    mode_set_pending_ = false;
    if (!isHoverEnabled()) {
      if (mission_->phase() == Phase::Idle) {
        RCLCPP_INFO(
          getNodePtr()->get_logger(),
          "TRAJECTORY mode set: the %s mission starts once motor speeds arrive",
          toString(mission_->config().type));
      } else {
        RCLCPP_INFO(
          getNodePtr()->get_logger(),
          "TRAJECTORY mode set in phase %s: no-op, the mission carries on",
          toString(mission_->phase()));
      }
    }
  }
  // The course is configuration, so a state is all a step needs.
  setReferenceReceived(true);
}

void Plugin::onUpdateReference(const geometry_msgs::msg::PoseStamped & ref)
{
  if (ready_ && hover_reference_armed_ && isHoverEnabled()) {
    if (ref.header.frame_id != getDesiredPoseFrameId()) {
      RCLCPP_ERROR(
        getNodePtr()->get_logger(), "HOVER reference in frame '%s', expected '%s': ignored",
        ref.header.frame_id.c_str(), getDesiredPoseFrameId().c_str());
      return;
    }
    const auto & position = ref.pose.position;
    const auto & orientation = ref.pose.orientation;
    const double yaw = yawFromQuaternion(
      Eigen::Quaterniond(orientation.w, orientation.x, orientation.y, orientation.z));
    if (!mission_->requestHold(position.x, position.y, position.z, yaw)) {
      RCLCPP_ERROR(
        getNodePtr()->get_logger(), "HOVER reference is not finite, waiting for the next one");
      return;
    }
    hover_reference_armed_ = false;
    handleMissionEvents();
    return;
  }
  RCLCPP_INFO_ONCE(
    getNodePtr()->get_logger(),
    "Pose references are ignored: the mission and the hold come from the plugin itself");
}

void Plugin::onUpdateReference(const geometry_msgs::msg::TwistStamped & ref)
{
  (void)ref;
  RCLCPP_INFO_ONCE(getNodePtr()->get_logger(), "Twist references are ignored");
}

void Plugin::onUpdateReference(const as2_msgs::msg::TrajectorySetpoints & ref)
{
  (void)ref;
  RCLCPP_INFO_ONCE(
    getNodePtr()->get_logger(),
    "Trajectory references are ignored: the course comes from the configuration");
}

void Plugin::onUpdateReference(const as2_msgs::msg::Thrust & ref)
{
  (void)ref;
  RCLCPP_INFO_ONCE(getNodePtr()->get_logger(), "Thrust references are ignored");
}

bool Plugin::computeOutput(
  double dt,
  geometry_msgs::msg::PoseStamped & pose,
  geometry_msgs::msg::TwistStamped & twist,
  as2_msgs::msg::Thrust & thrust)
{
  (void)dt;
  (void)pose;
  if (!ready_) {
    return false;
  }
  const rclcpp::Logger logger = getNodePtr()->get_logger();
  auto & clock = *getNodePtr()->get_clock();
  if (!motor_speed_received_) {
    RCLCPP_INFO_THROTTLE(
      logger, clock, 2000, "Waiting for motor speeds on '%s' before the first policy step",
      motor_speed_topic_.c_str());
    return false;
  }
  if (!state_valid_ || !isFinite(state_)) {
    RCLCPP_ERROR_THROTTLE(logger, clock, 1000, "The state is not finite, no command computed");
    return false;
  }
  if (mission_->phase() == Phase::Idle && isHoverEnabled()) {
    RCLCPP_WARN_THROTTLE(
      logger, clock, 1000, "HOVER mode without a hold setpoint yet, no command computed");
    return false;
  }

  const rclcpp::Time now = getNodePtr()->now();
  if (has_held_command_) {
    const double since_step = (now - last_step_time_).seconds();
    if (since_step >= 0.0 && since_step < 0.5 * dt_) {
      writeCommand(held_command_, now, twist, thrust);
      return true;
    }
    if (since_step > 1.5 * dt_) {
      ++overruns_;
      RCLCPP_WARN_THROTTLE(
        logger, clock, 1000, "Policy step overrun: %.4f s since the last step, dt %.4f s",
        since_step, dt_);
    }
  }

  motor_speed_age_s_ = (now - motor_speed_stamp_).seconds();
  const bool stale = motor_speed_age_s_ > motor_speed_timeout_s_;
  if (stale) {
    ++stale_steps_;
    RCLCPP_WARN_THROTTLE(
      logger, clock, 1000, "Motor speeds are %.3f s old (timeout %.3f s): using the last ones",
      motor_speed_age_s_, motor_speed_timeout_s_);
  }
  const bool stale_changed = stale != motor_speed_stale_;
  motor_speed_stale_ = stale;

  StepResult result;
  try {
    if (mission_->phase() == Phase::Idle) {
      mission_->start();
      handleMissionEvents();
    }
    result = mission_->step(state_, now.seconds());
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(logger, clock, 1000, "Policy step failed: %s", e.what());
    return false;
  }
  handleMissionEvents();
  if (stale_changed) {
    publishStatus();
  }

  held_command_ = result.command;
  has_held_command_ = true;
  last_step_time_ = now;
  writeCommand(result.command, now, twist, thrust);
  publishStep(result);
  return true;
}

void Plugin::motorSpeedCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (msg->velocity.size() < state_.motor_speeds.size()) {
    RCLCPP_WARN_THROTTLE(
      getNodePtr()->get_logger(), *getNodePtr()->get_clock(), 5000,
      "Motor speeds on '%s' carry %zu velocities, the observation reads %zu",
      motor_speed_topic_.c_str(), msg->velocity.size(), state_.motor_speeds.size());
    return;
  }
  for (std::size_t motor = 0; motor < state_.motor_speeds.size(); ++motor) {
    state_.motor_speeds[motor] = msg->velocity[motor];
  }
  motor_speed_stamp_ = getNodePtr()->now();
  if (!motor_speed_received_) {
    motor_speed_received_ = true;
    RCLCPP_INFO(
      getNodePtr()->get_logger(), "Motor speeds received on '%s'",
      motor_speed_sub_->get_topic_name());
  }
}

void Plugin::handleMissionEvents()
{
  const std::vector<MissionEvent> events = mission_->takeEvents();
  for (const MissionEvent & event : events) {
    last_event_ = describe(event);
    RCLCPP_INFO(getNodePtr()->get_logger(), "%s", last_event_.c_str());
  }
  if (!events.empty()) {
    publishStatus();
  }
}

void Plugin::writeCommand(
  const Command & command, const rclcpp::Time & stamp,
  geometry_msgs::msg::TwistStamped & twist, as2_msgs::msg::Thrust & thrust) const
{
  const std::string & frame_id = getNodePtr()->getBaseFrameId();
  thrust.header.stamp = stamp;
  thrust.header.frame_id = frame_id;
  thrust.thrust = static_cast<float>(command.thrust_n);

  twist.header.stamp = stamp;
  twist.header.frame_id = frame_id;
  twist.twist.linear.x = 0.0;
  twist.twist.linear.y = 0.0;
  twist.twist.linear.z = 0.0;
  twist.twist.angular.x = command.rates.x();
  twist.twist.angular.y = command.rates.y();
  twist.twist.angular.z = command.rates.z();
}

void Plugin::publishStep(const StepResult & result)
{
  if (!step_pub_) {
    return;
  }
  std_msgs::msg::Float64MultiArray msg;
  msg.layout.dim.resize(1);
  msg.layout.dim[0].label = kStepLabel;
  msg.layout.dim[0].size = kStepColumns;
  msg.layout.dim[0].stride = kStepColumns;
  msg.layout.data_offset = 0;

  std::vector<double> & data = msg.data;
  data.reserve(kStepColumns);
  data.push_back(result.t);
  data.push_back(static_cast<double>(static_cast<int>(result.status.phase)));
  data.push_back(static_cast<double>(result.status.target));
  data.push_back(static_cast<double>(result.status.lap));
  data.push_back(result.frame.position.x());
  data.push_back(result.frame.position.y());
  data.push_back(result.frame.position.z());
  data.push_back(result.frame.yaw);
  for (int axis = 0; axis < 3; ++axis) {
    data.push_back(state_.position(axis));
  }
  for (int axis = 0; axis < 3; ++axis) {
    data.push_back(state_.velocity(axis));
  }
  data.push_back(state_.orientation.w());
  data.push_back(state_.orientation.x());
  data.push_back(state_.orientation.y());
  data.push_back(state_.orientation.z());
  for (int axis = 0; axis < 3; ++axis) {
    data.push_back(state_.body_rates(axis));
  }
  for (const double speed : state_.motor_speeds) {
    data.push_back(speed);
  }
  for (std::size_t i = 0; i < kStepObservationColumns; ++i) {
    data.push_back(
      static_cast<Eigen::Index>(i) < result.observation.size() ?
      result.observation(static_cast<Eigen::Index>(i)) :
      std::numeric_limits<double>::quiet_NaN());
  }
  for (const double value : result.action) {
    data.push_back(value);
  }
  data.push_back(result.command.thrust_n);
  for (int axis = 0; axis < 3; ++axis) {
    data.push_back(result.command.rates(axis));
  }
  step_pub_->publish(msg);
}

void Plugin::publishStatus()
{
  if (status_pub_) {
    status_pub_->publish(statusMessage());
  }
}

diagnostic_msgs::msg::DiagnosticStatus Plugin::statusMessage() const
{
  diagnostic_msgs::msg::DiagnosticStatus msg;
  msg.name = "rl_policy";
  msg.hardware_id = getNodePtr()->get_namespace();
  msg.values.push_back(keyValue("ready", text(ready_)));
  if (!ready_ || !mission_) {
    msg.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    std::string reasons;
    for (const std::string & reason : refusals_) {
      reasons += (reasons.empty() ? "" : "; ") + reason;
    }
    msg.message = "not ready: " + reasons;
    return msg;
  }

  const MissionStatus & status = mission_->status();
  const MissionConfig & config = mission_->config();
  msg.level = motor_speed_stale_ ? diagnostic_msgs::msg::DiagnosticStatus::WARN :
    diagnostic_msgs::msg::DiagnosticStatus::OK;
  char summary[160];
  if (status.phase == Phase::Race) {
    std::snprintf(
      summary, sizeof(summary), "race: gate %d, %d of %d laps, %d gates passed", status.target,
      status.lap, config.laps, status.gates_passed);
  } else if (status.phase == Phase::Hold) {
    std::snprintf(
      summary, sizeof(summary), "hold at [%.2f, %.2f, %.2f] (%s)", status.setpoint.position.x(),
      status.setpoint.position.y(), status.setpoint.position.z(), toString(status.exit_reason));
  } else {
    std::snprintf(summary, sizeof(summary), "idle");
  }
  msg.message = summary;

  const CrossingRecord & crossing = status.last_crossing;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  msg.values.push_back(keyValue("mission", toString(config.type)));
  msg.values.push_back(keyValue("phase", toString(status.phase)));
  msg.values.push_back(keyValue("policy", status.policy));
  msg.values.push_back(keyValue("target", std::to_string(status.target)));
  msg.values.push_back(keyValue("lap", std::to_string(status.lap)));
  msg.values.push_back(keyValue("laps", std::to_string(config.laps)));
  msg.values.push_back(keyValue("gates_passed", std::to_string(status.gates_passed)));
  msg.values.push_back(
    keyValue(
      "last_crossing_gate",
      crossing.available ? std::to_string(crossing.gate) : std::string("-1")));
  msg.values.push_back(
    keyValue(
      "last_crossing_lateral_m", text(crossing.available ? crossing.crossing.lateral : nan)));
  msg.values.push_back(
    keyValue(
      "last_crossing_vertical_m", text(crossing.available ? crossing.crossing.vertical : nan)));
  msg.values.push_back(keyValue("last_crossing_passed", text(crossing.passed)));
  msg.values.push_back(keyValue("last_crossing_valid", text(crossing.valid)));
  msg.values.push_back(keyValue("valid_half_m", text(config.valid_half_m)));
  msg.values.push_back(keyValue("exit_reason", toString(status.exit_reason)));
  Gate setpoint{Eigen::Vector3d::Constant(nan), nan};
  if (status.holding) {
    setpoint = status.setpoint;
  }
  msg.values.push_back(keyValue("setpoint_x", text(setpoint.position.x())));
  msg.values.push_back(keyValue("setpoint_y", text(setpoint.position.y())));
  msg.values.push_back(keyValue("setpoint_z", text(setpoint.position.z())));
  msg.values.push_back(keyValue("setpoint_yaw", text(setpoint.yaw)));
  msg.values.push_back(keyValue("steps", std::to_string(status.steps)));
  msg.values.push_back(keyValue("overruns", std::to_string(overruns_)));
  const double motor_speed_age_s = motor_speed_received_ ?
    (getNodePtr()->now() - motor_speed_stamp_).seconds() : nan;
  msg.values.push_back(keyValue("motor_speed_age_s", text(motor_speed_age_s)));
  msg.values.push_back(keyValue("motor_speed_stale", text(motor_speed_stale_)));
  msg.values.push_back(keyValue("motor_speed_stale_steps", std::to_string(stale_steps_)));
  msg.values.push_back(keyValue("last_event", last_event_));
  return msg;
}

}  // namespace rl_policy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  rl_policy::Plugin,
  as2_motion_controller_plugin_base::ControllerBase)
