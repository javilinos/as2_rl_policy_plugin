#include "as2_rl_policy/core/gates.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rl_policy
{

Crossing planeCrossing(
  const Eigen::Vector3d & previous, const Eigen::Vector3d & current, const Gate & gate)
{
  const double normal_x = std::cos(gate.yaw);
  const double normal_y = std::sin(gate.yaw);
  const double proj_prev = (previous.x() - gate.position.x()) * normal_x +
    (previous.y() - gate.position.y()) * normal_y;
  const double proj_cur = (current.x() - gate.position.x()) * normal_x +
    (current.y() - gate.position.y()) * normal_y;

  Crossing crossing;
  if (!(proj_prev < 0.0 && proj_cur > 0.0)) {
    return crossing;
  }
  crossing.crossed = true;
  const double weight = proj_prev / (proj_prev - proj_cur);
  const Eigen::Vector3d hit = (previous - gate.position) + weight * (current - previous);
  crossing.lateral = -hit.x() * normal_y + hit.y() * normal_x;
  crossing.vertical = hit.z();
  return crossing;
}

GateSequencer::GateSequencer(Course course, SequencerConfig config)
: course_(std::move(course)), config_(config)
{
  if (config_.laps < 1) {
    throw std::invalid_argument("race.laps must be at least 1");
  }
  if (!(config_.pass_tolerance_m > 0.0) || !(config_.valid_half_m > 0.0)) {
    throw std::invalid_argument("race.pass_tolerance_m and race.valid_half_m must be positive");
  }
  if (!(config_.gate_timeout_s > 0.0) || !(config_.dt > 0.0)) {
    throw std::invalid_argument("race.gate_timeout_s and the policy dt must be positive");
  }
}

void GateSequencer::reset()
{
  target_ = 0;
  lap_ = 0;
  gates_passed_ = 0;
  steps_since_pass_ = 0;
  finished_ = false;
}

GateUpdate GateSequencer::update(const Eigen::Vector3d & previous, const Eigen::Vector3d & current)
{
  GateUpdate update;
  update.gate = target_;
  if (finished_) {
    return update;
  }

  update.crossing = planeCrossing(previous, current, course_.gate(target_));
  if (update.crossing.crossed) {
    const double worst =
      std::max(std::abs(update.crossing.lateral), std::abs(update.crossing.vertical));
    update.valid = worst < config_.valid_half_m;
    if (!(worst < config_.pass_tolerance_m)) {
      update.event = GateEvent::Missed;
      return update;
    }
    ++gates_passed_;
    if (target_ == course_.size() - 1) {
      ++lap_;
    }
    target_ = course_.next(target_);
    steps_since_pass_ = 0;
    update.event = GateEvent::Passed;
    if (lap_ == config_.laps) {
      finished_ = true;
      update.event = GateEvent::Finished;
    }
    return update;
  }

  ++steps_since_pass_;
  if (static_cast<double>(steps_since_pass_) * config_.dt > config_.gate_timeout_s) {
    update.event = GateEvent::Timeout;
  }
  return update;
}

}  // namespace rl_policy
