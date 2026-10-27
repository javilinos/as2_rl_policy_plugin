#include "as2_rl_policy/core/observation.hpp"

#include <cmath>

#include "as2_rl_policy/core/angles.hpp"

namespace rl_policy
{

bool isFinite(const VehicleState & state)
{
  if (!state.position.allFinite() || !state.velocity.allFinite() ||
    !state.orientation.coeffs().allFinite() || !state.body_rates.allFinite())
  {
    return false;
  }
  for (const double speed : state.motor_speeds) {
    if (!std::isfinite(speed)) {
      return false;
    }
  }
  return true;
}

ActionHistory::ActionHistory(std::size_t length)
: actions_(length, Action{})
{
}

void ActionHistory::push(const Action & action)
{
  if (actions_.empty()) {
    return;
  }
  for (std::size_t age = actions_.size() - 1; age > 0; --age) {
    actions_[age] = actions_[age - 1];
  }
  actions_[0] = action;
}

void ActionHistory::zero()
{
  for (Action & action : actions_) {
    action.fill(0.0);
  }
}

GateRelativeObservation gateRelativeObservation(
  const VehicleState & state, const Gate & target, const Eigen::Vector4d & next_block,
  double motor_speed_normalization)
{
  GateRelativeObservation obs;
  const double c = std::cos(target.yaw);
  const double s = std::sin(target.yaw);

  const double dx = state.position.x() - target.position.x();
  const double dy = state.position.y() - target.position.y();
  obs(0) = c * dx + s * dy;
  obs(1) = -s * dx + c * dy;
  obs(2) = state.position.z() - target.position.z();
  obs(3) = c * state.velocity.x() + s * state.velocity.y();
  obs(4) = -s * state.velocity.x() + c * state.velocity.y();
  obs(5) = state.velocity.z();

  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  quaternionToEuler(state.orientation, roll, pitch, yaw);
  obs(6) = wrapPi(roll);
  obs(7) = wrapPi(pitch);
  obs(8) = wrapPi(yaw - target.yaw);

  obs(9) = state.body_rates.x();
  obs(10) = state.body_rates.y();
  obs(11) = state.body_rates.z();
  for (int motor = 0; motor < 4; ++motor) {
    obs(12 + motor) = 2.0 * state.motor_speeds[motor] / motor_speed_normalization - 1.0;
  }

  obs.segment<4>(16) = next_block;
  return obs;
}

Eigen::Matrix<double, 6, 1> r6FromEuler(double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  Eigen::Matrix<double, 6, 1> r6;
  r6 << cy * cp, sy * cp, -sp, cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr;
  return r6;
}

Eigen::VectorXd gateRelativeBlock(
  const VehicleState & state, const Gate & target, const Eigen::Vector4d & next_block,
  double motor_speed_normalization, Attitude attitude)
{
  const GateRelativeObservation kernel =
    gateRelativeObservation(state, target, next_block, motor_speed_normalization);
  if (attitude == Attitude::Euler) {
    return kernel;
  }
  const auto f32 = [](double value) {return static_cast<double>(static_cast<float>(value));};
  Eigen::VectorXd block(kGateRelativeR6Size);
  block.head<6>() = kernel.head<6>();
  block.segment<6>(6) = r6FromEuler(f32(kernel(6)), f32(kernel(7)), f32(kernel(8)));
  block.tail<kGateRelativeSize - 9>() = kernel.tail<kGateRelativeSize - 9>();
  return block;
}

Eigen::VectorXd buildObservation(
  const VehicleState & state, const Course & course, std::size_t target,
  const ActionHistory & history, double motor_speed_normalization, Attitude attitude)
{
  const Eigen::Index base = gateRelativeSize(attitude);
  Eigen::VectorXd obs(base + kActionSize * static_cast<Eigen::Index>(history.size()));
  obs.head(base) = gateRelativeBlock(
    state, course.gate(target), course.nextBlock(target), motor_speed_normalization, attitude);
  for (std::size_t age = 0; age < history.size(); ++age) {
    for (int k = 0; k < kActionSize; ++k) {
      obs(base + kActionSize * static_cast<Eigen::Index>(age) + k) = history[age][k];
    }
  }
  return obs.cast<float>().cast<double>();
}

}  // namespace rl_policy
