#ifndef AS2_RL_POLICY__CORE__OBSERVATION_HPP_
#define AS2_RL_POLICY__CORE__OBSERVATION_HPP_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <cstddef>
#include <vector>

#include "as2_rl_policy/core/course.hpp"
#include "as2_rl_policy/core/policy.hpp"

namespace rl_policy
{

struct VehicleState
{
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  // World frame (ENU), m/s.
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  // Body frame (FLU), rad/s.
  Eigen::Vector3d body_rates = Eigen::Vector3d::Zero();
  // Plant order, rad/s.
  std::array<double, 4> motor_speeds{};
};

bool isFinite(const VehicleState & state);

// The applied actions, newest first.
class ActionHistory
{
public:
  explicit ActionHistory(std::size_t length = 0);

  void push(const Action & action);

  void zero();

  std::size_t size() const {return actions_.size();}

  const Action & operator[](std::size_t age) const {return actions_[age];}

private:
  std::vector<Action> actions_;
};

using GateRelativeObservation = Eigen::Matrix<double, kGateRelativeSize, 1>;

// The kernel's observe() entries 0..19, in double.
GateRelativeObservation gateRelativeObservation(
  const VehicleState & state, const Gate & target, const Eigen::Vector4d & next_block,
  double motor_speed_normalization);

// [R00, R10, R20, R02, R12, R22] of R = Rz(yaw) Ry(pitch) Rx(roll): the body x and thrust axes,
// quadlab.obs.r6_from_euler.
Eigen::Matrix<double, 6, 1> r6FromEuler(double roll, double pitch, double yaw);

// The gate-relative entries of a layout: the kernel's 0..19, or with R6 their three angles
// replaced by the R6 of those angles as float32, as the trainer encodes the kernel's float32 row
// (quadlab.envs.rates_cpp.encode_row): 0..5, the six R6 entries, then the kernel's 9..19.
Eigen::VectorXd gateRelativeBlock(
  const VehicleState & state, const Gate & target, const Eigen::Vector4d & next_block,
  double motor_speed_normalization, Attitude attitude);

// The gate-relative entries at the target of the course, then the history, rounded to float32
// as trained.
Eigen::VectorXd buildObservation(
  const VehicleState & state, const Course & course, std::size_t target,
  const ActionHistory & history, double motor_speed_normalization,
  Attitude attitude = Attitude::Euler);

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__OBSERVATION_HPP_
