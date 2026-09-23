#ifndef AS2_RL_POLICY__CORE__GATES_HPP_
#define AS2_RL_POLICY__CORE__GATES_HPP_

#include <Eigen/Dense>

#include <cstddef>

#include "as2_rl_policy/core/course.hpp"

namespace rl_policy
{

struct Crossing
{
  bool crossed = false;
  // Offsets of the piercing point from the gate centre, along [-sin yaw, cos yaw, 0] and world up.
  double lateral = 0.0;
  double vertical = 0.0;
};

// The segment previous -> current against the gate plane, crossed only from its approach side.
Crossing planeCrossing(
  const Eigen::Vector3d & previous, const Eigen::Vector3d & current, const Gate & gate);

struct SequencerConfig
{
  int laps = 0;
  double pass_tolerance_m = 0.0;
  double valid_half_m = 0.0;
  double gate_timeout_s = 0.0;
  double dt = 0.0;
};

enum class GateEvent
{
  None,
  Passed,
  Finished,
  Missed,
  Timeout,
};

struct GateUpdate
{
  GateEvent event = GateEvent::None;
  // The gate this update was checked against.
  std::size_t gate = 0;
  Crossing crossing;
  bool valid = false;
};

class GateSequencer
{
public:
  GateSequencer(Course course, SequencerConfig config);

  void reset();

  GateUpdate update(const Eigen::Vector3d & previous, const Eigen::Vector3d & current);

  const Course & course() const {return course_;}

  const SequencerConfig & config() const {return config_;}

  std::size_t target() const {return target_;}

  int lap() const {return lap_;}

  int gatesPassed() const {return gates_passed_;}

  int stepsSincePass() const {return steps_since_pass_;}

  bool finished() const {return finished_;}

private:
  Course course_;
  SequencerConfig config_;
  std::size_t target_ = 0;
  int lap_ = 0;
  int gates_passed_ = 0;
  int steps_since_pass_ = 0;
  bool finished_ = false;
};

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__GATES_HPP_
