#ifndef AS2_RL_POLICY__CORE__MISSION_HPP_
#define AS2_RL_POLICY__CORE__MISSION_HPP_

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "as2_rl_policy/core/course.hpp"
#include "as2_rl_policy/core/decode.hpp"
#include "as2_rl_policy/core/gates.hpp"
#include "as2_rl_policy/core/observation.hpp"
#include "as2_rl_policy/core/policy.hpp"

namespace rl_policy
{

inline constexpr const char kRacePolicy[] = "race";
inline constexpr const char kHoverPolicy[] = "hover";

enum class Phase
{
  Idle = 0,
  Race = 1,
  Hold = 2,
};

enum class MissionType
{
  Race,
  Hover,
};

enum class ExitReason
{
  None,
  Finished,
  Missed,
  Timeout,
  OutOfBounds,
  HoverRequest,
};

// Where the race's finish holds: the pose it ends at, or the hold setpoint.
enum class AfterRace
{
  Here,
  Setpoint,
};

const char * toString(Phase phase);

const char * toString(MissionType type);

const char * toString(ExitReason reason);

const char * toString(AfterRace after_race);

// Throws std::invalid_argument for anything but "race" or "hover".
MissionType missionTypeFromString(const std::string & name);

// Throws std::invalid_argument for anything but "here" or "setpoint".
AfterRace afterRaceFromString(const std::string & name);

struct MissionConfig
{
  MissionType type = MissionType::Race;
  std::vector<Gate> gates;
  Bounds bounds;
  int laps = 0;
  double pass_tolerance_m = 0.0;
  double valid_half_m = 0.0;
  // Unset, the square's values.
  std::optional<double> octagon_pass_tolerance_m;
  std::optional<double> octagon_valid_half_m;
  double gate_timeout_s = 0.0;
  double bounds_margin_m = 0.0;
  double ceiling_m = 0.0;
  int exit_debounce_steps = 0;
  Gate hold_setpoint;
  AfterRace after_race = AfterRace::Here;
  double min_altitude_m = 0.0;
  double mass_kg = 0.0;
};

struct CrossingRecord
{
  bool available = false;
  std::size_t gate = 0;
  Crossing crossing;
  bool passed = false;
  bool valid = false;
};

struct MissionStatus
{
  Phase phase = Phase::Idle;
  std::string policy;
  // The target gate in Race, -1 otherwise.
  int target = -1;
  int lap = 0;
  int gates_passed = 0;
  CrossingRecord last_crossing;
  ExitReason exit_reason = ExitReason::None;
  bool holding = false;
  Gate setpoint;
  std::uint64_t steps = 0;
};

struct MissionEvent
{
  enum class Type
  {
    PhaseChange,
    GatePassed,
  };

  Type type = Type::PhaseChange;
  Phase from = Phase::Idle;
  Phase to = Phase::Idle;
  ExitReason reason = ExitReason::None;
  std::string policy;
  std::size_t gate = 0;
  int lap = 0;
  int gates_passed = 0;
  Crossing crossing;
  bool valid = false;
  double since_pass_s = 0.0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Gate setpoint;
};

std::string describe(const MissionEvent & event);

struct StepResult
{
  double t = 0.0;
  Command command;
  Action action{};
  Eigen::VectorXd observation;
  // The target frame the observation is relative to: the gate, or the hold setpoint.
  Gate frame;
  MissionStatus status;
};

class MissionController
{
public:
  MissionController(MissionConfig config, PolicyBank policies);

  // Idle only: the race mission races from gate 0, the hover mission holds its setpoint.
  bool start();

  // A HOVER request from any phase: hold at the clamped pose with the hover policy.
  bool requestHold(double x, double y, double z, double yaw);

  // One policy step, exactly one act() of the active policy.
  StepResult step(const VehicleState & state, double t);

  // (x, y) into the bounds, z no lower than the minimum altitude, yaw unchanged.
  Gate clampSetpoint(double x, double y, double z, double yaw) const;

  std::vector<MissionEvent> takeEvents();

  Phase phase() const {return status_.phase;}

  const MissionStatus & status() const {return status_;}

  const MissionConfig & config() const {return config_;}

  const PolicyBank & policies() const {return policies_;}

  double dt() const {return policies_.dt();}

  const ActionHistory & history() const {return history_;}

  const GateSequencer & sequencer() const {return sequencer_;}

private:
  static MissionConfig validated(MissionConfig config, const PolicyBank & policies);

  void activate(const std::string & name);

  void raceChecks(const VehicleState & state);

  void exitRace(ExitReason reason, const VehicleState & state, const GateUpdate & update);

  void hold(const Gate & setpoint, ExitReason reason, MissionEvent event);

  MissionConfig config_;
  PolicyBank policies_;
  GateSequencer sequencer_;
  Course hold_course_;
  std::shared_ptr<const Policy> active_;
  ActionHistory history_;
  MissionStatus status_;
  Eigen::Vector3d previous_position_ = Eigen::Vector3d::Zero();
  bool has_previous_position_ = false;
  int outside_steps_ = 0;
  std::vector<MissionEvent> events_;
};

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__MISSION_HPP_
