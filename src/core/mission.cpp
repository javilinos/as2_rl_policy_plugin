#include "as2_rl_policy/core/mission.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "as2_rl_policy/core/angles.hpp"

namespace rl_policy
{

namespace
{

bool finite(const Gate & gate)
{
  return gate.position.allFinite() && std::isfinite(gate.yaw);
}

bool samePose(const Gate & a, const Gate & b)
{
  return a.position == b.position && a.yaw == b.yaw;
}

void require(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

std::string format(const char * pattern, ...)
{
  char buffer[512];
  va_list args;
  va_start(args, pattern);
  std::vsnprintf(buffer, sizeof(buffer), pattern, args);
  va_end(args);
  return buffer;
}

}  // namespace

const char * toString(Phase phase)
{
  switch (phase) {
    case Phase::Idle:
      return "idle";
    case Phase::Race:
      return "race";
    case Phase::Hold:
      return "hold";
  }
  return "unknown";
}

const char * toString(MissionType type)
{
  switch (type) {
    case MissionType::Race:
      return "race";
    case MissionType::Hover:
      return "hover";
  }
  return "unknown";
}

const char * toString(ExitReason reason)
{
  switch (reason) {
    case ExitReason::None:
      return "none";
    case ExitReason::Finished:
      return "finished";
    case ExitReason::Missed:
      return "missed";
    case ExitReason::Timeout:
      return "timeout";
    case ExitReason::OutOfBounds:
      return "out_of_bounds";
    case ExitReason::HoverRequest:
      return "hover_request";
  }
  return "unknown";
}

const char * toString(AfterRace after_race)
{
  switch (after_race) {
    case AfterRace::Here:
      return "here";
    case AfterRace::Setpoint:
      return "setpoint";
  }
  return "unknown";
}

const char * toString(HoldReference reference)
{
  switch (reference) {
    case HoldReference::SentOrConfigured:
      return "sent_or_configured";
    case HoldReference::Configured:
      return "configured";
  }
  return "unknown";
}

const char * toString(SetpointSource source)
{
  switch (source) {
    case SetpointSource::None:
      return "none";
    case SetpointSource::Configured:
      return "configured";
    case SetpointSource::Sent:
      return "sent";
    case SetpointSource::Here:
      return "here";
  }
  return "unknown";
}

MissionType missionTypeFromString(const std::string & name)
{
  if (name == toString(MissionType::Race)) {
    return MissionType::Race;
  }
  if (name == toString(MissionType::Hover)) {
    return MissionType::Hover;
  }
  throw std::invalid_argument("mission '" + name + "' is neither 'race' nor 'hover'");
}

AfterRace afterRaceFromString(const std::string & name)
{
  if (name == toString(AfterRace::Here)) {
    return AfterRace::Here;
  }
  if (name == toString(AfterRace::Setpoint)) {
    return AfterRace::Setpoint;
  }
  throw std::invalid_argument("after_race '" + name + "' is neither 'here' nor 'setpoint'");
}

HoldReference holdReferenceFromString(const std::string & name)
{
  if (name == toString(HoldReference::SentOrConfigured)) {
    return HoldReference::SentOrConfigured;
  }
  if (name == toString(HoldReference::Configured)) {
    return HoldReference::Configured;
  }
  throw std::invalid_argument(
          "reference '" + name + "' is neither 'sent_or_configured' nor 'configured'");
}

std::string describe(const MissionEvent & event)
{
  if (event.type == MissionEvent::Type::GatePassed) {
    return format(
      "Gate %zu passed: lateral %.3f m, vertical %.3f m, %s; %d gates passed, %d laps completed",
      event.gate, event.crossing.lateral, event.crossing.vertical,
      event.valid ? "valid" : "outside the valid opening", event.gates_passed, event.lap);
  }
  if (event.type == MissionEvent::Type::SetpointChange) {
    return format(
      "Sent reference: the hold %s [%.3f, %.3f, %.3f] yaw %.3f rad",
      event.from == Phase::Hold ? "moves to" : "starts at", event.setpoint.position.x(),
      event.setpoint.position.y(), event.setpoint.position.z(), event.setpoint.yaw);
  }
  if (event.to == Phase::Race) {
    return format("Race started from gate %zu with policy '%s'", event.gate, event.policy.c_str());
  }
  std::string cause;
  switch (event.reason) {
    case ExitReason::None:
      cause = event.source == SetpointSource::Sent ? "Holding the sent reference" :
        "Holding the configured setpoint";
      break;
    case ExitReason::Finished:
      cause = format("Race finished: %d laps, %d gates", event.lap, event.gates_passed);
      break;
    case ExitReason::Missed:
      cause = format(
        "Gate %zu missed: crossed its plane at lateral %.3f m, vertical %.3f m", event.gate,
        event.crossing.lateral, event.crossing.vertical);
      break;
    case ExitReason::Timeout:
      cause = format("Gate %zu not passed in %.2f s", event.gate, event.since_pass_s);
      break;
    case ExitReason::OutOfBounds:
      cause = format(
        "Out of bounds at [%.2f, %.2f, %.2f]", event.position.x(), event.position.y(),
        event.position.z());
      break;
    case ExitReason::HoverRequest:
      cause = "HOVER request";
      break;
  }
  return cause + format(
    " (%s -> %s): hold at [%.3f, %.3f, %.3f] yaw %.3f rad with policy '%s'",
    toString(event.from), toString(event.to), event.setpoint.position.x(),
    event.setpoint.position.y(), event.setpoint.position.z(), event.setpoint.yaw,
    event.policy.c_str());
}

MissionConfig MissionController::validated(MissionConfig config, const PolicyBank & policies)
{
  require(policies.size() > 0, "no policy is loaded");
  require(policies.has(kHoverPolicy), "a policy named 'hover' is required: every mission can hold");
  if (config.type == MissionType::Race) {
    require(policies.has(kRacePolicy), "the race mission needs a policy named 'race'");
  }
  require(!config.gates.empty(), "the course needs at least one gate");
  for (const Gate & gate : config.gates) {
    require(finite(gate), "the course has a non-finite gate");
  }
  const Bounds & bounds = config.bounds;
  require(
    std::isfinite(bounds.x_min) && std::isfinite(bounds.x_max) && std::isfinite(bounds.y_min) &&
    std::isfinite(bounds.y_max), "the course bounds must be finite");
  require(
    bounds.x_min < bounds.x_max && bounds.y_min < bounds.y_max,
    "the course bounds must be [min, max] with min < max");
  require(config.laps >= 1, "race.laps must be at least 1");
  require(config.pass_tolerance_m > 0.0, "race.pass_tolerance_m must be positive");
  require(config.valid_half_m > 0.0, "race.valid_half_m must be positive");
  require(
    !config.octagon_pass_tolerance_m || *config.octagon_pass_tolerance_m > 0.0,
    "race.octagon_pass_tolerance_m must be positive");
  require(
    !config.octagon_valid_half_m || *config.octagon_valid_half_m > 0.0,
    "race.octagon_valid_half_m must be positive");
  require(config.gate_timeout_s > 0.0, "race.gate_timeout_s must be positive");
  require(config.bounds_margin_m >= 0.0, "race.bounds_margin_m must not be negative");
  require(std::isfinite(config.ceiling_m), "race.ceiling_m must be finite");
  require(config.exit_debounce_steps >= 1, "race.exit_debounce_steps must be at least 1");
  require(finite(config.hold_setpoint), "hold.setpoint must be finite");
  if (config.type == MissionType::Race && config.after_race == AfterRace::Setpoint) {
    require(
      bounds.contains(config.hold_setpoint.position.x(), config.hold_setpoint.position.y()),
      "hold.after_race is setpoint, so hold.setpoint must lie inside the course bounds in x and y");
  }
  require(std::isfinite(config.min_altitude_m), "hold.min_altitude_m must be finite");
  require(
    config.mass_kg > 0.0 && std::isfinite(config.mass_kg), "vehicle.mass_kg must be positive");
  return config;
}

MissionController::MissionController(MissionConfig config, PolicyBank policies)
: config_(validated(std::move(config), policies)),
  policies_(std::move(policies)),
  sequencer_(
    Course(config_.gates, config_.bounds),
    SequencerConfig{config_.laps, config_.pass_tolerance_m, config_.valid_half_m,
      config_.gate_timeout_s, policies_.dt(), config_.octagon_pass_tolerance_m,
      config_.octagon_valid_half_m}),
  hold_course_(Course::single(config_.hold_setpoint))
{
  if (config_.type == MissionType::Hover) {
    const Gate & setpoint = config_.hold_setpoint;
    status_.setpoint = clampSetpoint(
      setpoint.position.x(), setpoint.position.y(), setpoint.position.z(), setpoint.yaw);
    status_.setpoint_source = SetpointSource::Configured;
  }
}

bool MissionController::start()
{
  if (status_.phase != Phase::Idle) {
    return false;
  }
  if (config_.type == MissionType::Hover) {
    const Gate setpoint = status_.setpoint;
    hold(setpoint, status_.setpoint_source, ExitReason::None, MissionEvent());
    return true;
  }

  sequencer_.reset();
  has_previous_position_ = false;
  outside_steps_ = 0;
  status_.phase = Phase::Race;
  status_.target = static_cast<int>(sequencer_.target());
  status_.lap = 0;
  status_.gates_passed = 0;
  status_.exit_reason = ExitReason::None;
  status_.holding = false;
  activate(kRacePolicy);

  MissionEvent event;
  event.type = MissionEvent::Type::PhaseChange;
  event.from = Phase::Idle;
  event.to = Phase::Race;
  event.policy = status_.policy;
  event.gate = sequencer_.target();
  events_.push_back(event);
  return true;
}

bool MissionController::requestHold(double x, double y, double z, double yaw)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(yaw)) {
    return false;
  }
  MissionEvent event;
  event.gate = sequencer_.target();
  hold(clampSetpoint(x, y, z, yaw), SetpointSource::Here, ExitReason::HoverRequest, event);
  return true;
}

ReferenceUpdate MissionController::updateHoldReference(double x, double y, double z, double yaw)
{
  if (config_.type != MissionType::Hover) {
    return ReferenceUpdate::IgnoredRace;
  }
  if (config_.hold_reference == HoldReference::Configured) {
    return ReferenceUpdate::IgnoredConfigured;
  }
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(yaw)) {
    return ReferenceUpdate::NotFinite;
  }
  Gate reference;
  reference.position = Eigen::Vector3d(x, y, z);
  reference.yaw = yaw;
  sent_reference_ = reference;
  const Gate setpoint = clampSetpoint(x, y, z, yaw);
  if (status_.setpoint_source == SetpointSource::Sent && samePose(setpoint, status_.setpoint)) {
    return ReferenceUpdate::Unchanged;
  }

  status_.setpoint = setpoint;
  status_.setpoint_source = SetpointSource::Sent;
  const bool holding = status_.phase == Phase::Hold;
  if (holding) {
    hold_course_ = Course::single(setpoint);
  }
  MissionEvent event;
  event.type = MissionEvent::Type::SetpointChange;
  event.from = status_.phase;
  event.to = status_.phase;
  event.reason = status_.exit_reason;
  event.policy = status_.policy;
  event.setpoint = setpoint;
  event.source = SetpointSource::Sent;
  events_.push_back(event);
  return holding ? ReferenceUpdate::Moved : ReferenceUpdate::Pending;
}

StepResult MissionController::step(const VehicleState & state, double t)
{
  if (status_.phase == Phase::Idle) {
    throw std::logic_error("MissionController::step() called before start()");
  }
  if (status_.phase == Phase::Race) {
    raceChecks(state);
  }

  const bool racing = status_.phase == Phase::Race;
  const Course & course = racing ? sequencer_.course() : hold_course_;
  const std::size_t target = racing ? sequencer_.target() : 0;
  const PolicySpec & spec = active_->spec();

  StepResult result;
  result.t = t;
  result.frame = course.gate(target);
  result.observation = buildObservation(
    state, course, target, history_, spec.observation.motor_speed_normalization);
  result.action = clip(active_->act(result.observation));
  history_.push(result.action);
  result.command = decode(result.action, spec.action, config_.mass_kg);

  status_.target = racing ? static_cast<int>(target) : -1;
  ++status_.steps;
  result.status = status_;
  return result;
}

Gate MissionController::clampSetpoint(double x, double y, double z, double yaw) const
{
  const Eigen::Vector2d xy = config_.bounds.clamp(x, y);
  Gate setpoint;
  setpoint.position = Eigen::Vector3d(xy.x(), xy.y(), std::max(z, config_.min_altitude_m));
  setpoint.yaw = yaw;
  return setpoint;
}

std::vector<MissionEvent> MissionController::takeEvents()
{
  std::vector<MissionEvent> events;
  events.swap(events_);
  return events;
}

void MissionController::activate(const std::string & name)
{
  if (active_ && status_.policy == name) {
    return;
  }
  active_ = policies_.get(name);
  status_.policy = name;
  history_ = ActionHistory(static_cast<std::size_t>(active_->spec().observation.action_history));
}

void MissionController::raceChecks(const VehicleState & state)
{
  const Eigen::Vector3d current = state.position;
  if (has_previous_position_) {
    const GateUpdate update = sequencer_.update(previous_position_, current);
    status_.target = static_cast<int>(sequencer_.target());
    status_.lap = sequencer_.lap();
    status_.gates_passed = sequencer_.gatesPassed();
    const bool passed =
      update.event == GateEvent::Passed || update.event == GateEvent::Finished;
    if (update.crossing.crossed) {
      status_.last_crossing = CrossingRecord{true, update.gate, update.crossing, passed,
        update.valid};
    }
    if (passed) {
      MissionEvent event;
      event.type = MissionEvent::Type::GatePassed;
      event.from = Phase::Race;
      event.to = Phase::Race;
      event.policy = status_.policy;
      event.gate = update.gate;
      event.lap = status_.lap;
      event.gates_passed = status_.gates_passed;
      event.crossing = update.crossing;
      event.valid = update.valid;
      events_.push_back(event);
    }
    switch (update.event) {
      case GateEvent::Finished:
        exitRace(ExitReason::Finished, state, update);
        break;
      case GateEvent::Missed:
        exitRace(ExitReason::Missed, state, update);
        break;
      case GateEvent::Timeout:
        exitRace(ExitReason::Timeout, state, update);
        break;
      case GateEvent::None:
      case GateEvent::Passed:
        break;
    }
  }
  previous_position_ = current;
  has_previous_position_ = true;
  if (status_.phase != Phase::Race) {
    return;
  }

  const bool outside =
    !config_.bounds.contains(current.x(), current.y(), config_.bounds_margin_m) ||
    current.z() > config_.ceiling_m;
  outside_steps_ = outside ? outside_steps_ + 1 : 0;
  if (outside_steps_ >= config_.exit_debounce_steps) {
    GateUpdate update;
    update.gate = sequencer_.target();
    exitRace(ExitReason::OutOfBounds, state, update);
  }
}

void MissionController::exitRace(
  ExitReason reason, const VehicleState & state, const GateUpdate & update)
{
  MissionEvent event;
  event.gate = update.gate;
  event.crossing = update.crossing;
  event.valid = update.valid;
  event.since_pass_s = static_cast<double>(sequencer_.stepsSincePass()) * dt();
  event.position = state.position;
  const bool to_setpoint =
    reason == ExitReason::Finished && config_.after_race == AfterRace::Setpoint;
  const Gate pose = to_setpoint ? config_.hold_setpoint :
    Gate{state.position, yawFromQuaternion(state.orientation)};
  hold(
    clampSetpoint(pose.position.x(), pose.position.y(), pose.position.z(), pose.yaw),
    to_setpoint ? SetpointSource::Configured : SetpointSource::Here, reason, event);
}

void MissionController::hold(
  const Gate & setpoint, SetpointSource source, ExitReason reason, MissionEvent event)
{
  event.type = MissionEvent::Type::PhaseChange;
  event.from = status_.phase;
  event.to = Phase::Hold;
  event.reason = reason;
  event.setpoint = setpoint;
  event.source = source;
  event.lap = status_.lap;
  event.gates_passed = status_.gates_passed;

  hold_course_ = Course::single(setpoint);
  status_.phase = Phase::Hold;
  status_.target = -1;
  status_.exit_reason = reason;
  status_.holding = true;
  status_.setpoint = setpoint;
  status_.setpoint_source = source;
  activate(kHoverPolicy);

  event.policy = status_.policy;
  events_.push_back(event);
}

}  // namespace rl_policy
