#ifndef AS2_RL_POLICY__CORE__GATES_HPP_
#define AS2_RL_POLICY__CORE__GATES_HPP_

#include <Eigen/Dense>

#include <cstddef>
#include <optional>

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

// max(|lateral|, |vertical|); an octagon's also takes (|u| + |v|) / sqrt(2).
double openingNorm(GateShape shape, double lateral, double vertical);

const char * toString(GateShape shape);

// How many of the course's gates have this shape.
std::size_t countShape(const Course & course, GateShape shape);

// How many of the course's gates are virtual.
std::size_t countVirtual(const Course & course);

struct SequencerConfig
{
  int laps = 0;
  // The pass windows, a shape's for every gate of it, virtual or not.
  double pass_tolerance_m = 0.0;
  double valid_half_m = 0.0;
  double gate_timeout_s = 0.0;
  double dt = 0.0;
  // Unset, an octagon is held to the square's window and valid window, in its own norm.
  std::optional<double> octagon_pass_tolerance_m;
  std::optional<double> octagon_valid_half_m;
  // A virtual gate's valid window, logged only; unset, its shape's.
  std::optional<double> virtual_valid_half_m;

  double passTolerance(GateShape shape) const;

  double validHalf(GateShape shape) const;

  double validHalf(const Gate & gate) const;
};

enum class GateEvent
{
  None,
  Passed,
  Finished,
  Missed,
  Timeout,
  // A virtual gate's plane crossed at or outside its pass window: no pass, no miss, the target
  // stays.
  Outside,
};

struct GateUpdate
{
  GateEvent event = GateEvent::None;
  // The gate this update was checked against.
  std::size_t gate = 0;
  Crossing crossing;
  // Inside the valid window: logged, never what decides a pass.
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
