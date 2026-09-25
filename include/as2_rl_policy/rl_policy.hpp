#ifndef AS2_RL_POLICY__RL_POLICY_HPP_
#define AS2_RL_POLICY__RL_POLICY_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "as2_motion_controller/controller_base.hpp"
#include "as2_msgs/msg/control_mode.hpp"
#include "as2_msgs/msg/thrust.hpp"
#include "as2_msgs/msg/trajectory_setpoints.hpp"

#include "as2_rl_policy/core/mission.hpp"

namespace rl_policy
{

// Column layout of the step debug topic.
inline constexpr const char kStepLabel[] = "rl_policy_step_v1";
inline constexpr std::size_t kStepColumns = 61;
inline constexpr std::size_t kStepObservationColumns = 28;

class Plugin : public as2_motion_controller_plugin_base::ControllerBase
{
public:
  Plugin() = default;
  ~Plugin() override = default;

  void ownInitialize() override;

  void updateParameter(const std::string & name, const rclcpp::Parameter & parameter) override;

  std::vector<std::string> initParameters() const override;

  void reset() override;

  as2_msgs::msg::ControlMode hoverMode() const override;

  bool onSetMode(
    const as2_msgs::msg::ControlMode & mode_in,
    const as2_msgs::msg::ControlMode & mode_out) override;

  void onUpdateState(
    const geometry_msgs::msg::PoseStamped & pose_msg,
    const geometry_msgs::msg::TwistStamped & twist_msg) override;

  void onUpdateReference(const geometry_msgs::msg::PoseStamped & ref) override;

  void onUpdateReference(const geometry_msgs::msg::TwistStamped & ref) override;

  void onUpdateReference(const as2_msgs::msg::TrajectorySetpoints & ref) override;

  void onUpdateReference(const as2_msgs::msg::Thrust & ref) override;

  bool computeOutput(
    double dt,
    geometry_msgs::msg::PoseStamped & pose,
    geometry_msgs::msg::TwistStamped & twist,
    as2_msgs::msg::Thrust & thrust) override;

  bool isReady() const {return ready_;}

  // Null until the plugin is ready.
  const MissionController * mission() const {return mission_.get();}

  std::uint64_t overruns() const {return overruns_;}

private:
  struct Settings
  {
    MissionConfig mission;
    std::vector<std::string> policy_names;
    std::vector<std::string> policy_files;
    bool policies_named = false;
    std::string motor_speed_topic;
    double motor_speed_timeout_s = 0.0;
  };

  void readSettings(Settings & settings);

  const rclcpp::Parameter * setting(const std::string & tail, rclcpp::ParameterType type);

  // Null, and no refusal, when no file sets it.
  const rclcpp::Parameter * optionalSetting(const std::string & tail, rclcpp::ParameterType type);

  void refuse(const std::string & reason);

  void motorSpeedCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

  void handleMissionEvents();

  void writeCommand(
    const Command & command, const rclcpp::Time & stamp,
    geometry_msgs::msg::TwistStamped & twist, as2_msgs::msg::Thrust & thrust) const;

  void publishStep(const StepResult & result);

  void publishStatus();

  diagnostic_msgs::msg::DiagnosticStatus statusMessage() const;

  // Configuration, stored as delivered and validated in ownInitialize().
  std::map<std::string, rclcpp::Parameter> parameters_;
  std::vector<std::string> refusals_;
  bool ready_ = false;

  std::unique_ptr<MissionController> mission_;
  double dt_ = 0.0;
  std::string motor_speed_topic_;
  double motor_speed_timeout_s_ = 0.0;

  // Latest state, in the earth frame; body rates in the body frame.
  VehicleState state_;
  bool state_valid_ = false;
  bool motor_speed_received_ = false;
  rclcpp::Time motor_speed_stamp_{0, 0, RCL_ROS_TIME};
  double motor_speed_age_s_ = 0.0;
  bool motor_speed_stale_ = false;
  std::uint64_t stale_steps_ = 0;

  // The last policy step, replayed to a call that comes early.
  Command held_command_;
  bool has_held_command_ = false;
  rclcpp::Time last_step_time_{0, 0, RCL_ROS_TIME};
  std::uint64_t overruns_ = 0;
  std::string last_event_;

  // Armed by reset(): the next pose reference in a HOVER mode is the hold setpoint.
  bool hover_reference_armed_ = false;
  bool mode_set_pending_ = false;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr motor_speed_sub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr step_pub_;
  rclcpp::TimerBase::SharedPtr status_timer_;
};

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__RL_POLICY_HPP_
