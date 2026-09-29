#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "as2_rl_policy/core/mission.hpp"
#include "fixture_utils.hpp"

namespace
{

using rl_policy::Action;
using rl_policy::AfterRace;
using rl_policy::ExitReason;
using rl_policy::GateShape;
using rl_policy::HoldReference;
using rl_policy::MissionConfig;
using rl_policy::MissionController;
using rl_policy::MissionEvent;
using rl_policy::MissionType;
using rl_policy::Phase;
using rl_policy::PolicyBank;
using rl_policy::ReferenceUpdate;
using rl_policy::SetpointSource;
using rl_policy::StepResult;
using rl_policy::VehicleState;

constexpr double kDt = 0.01;

// Records every observation and answers with an action that changes on every call.
class CountingPolicy : public rl_policy::Policy
{
public:
  CountingPolicy(const std::string & task, double offset)
  : offset_(offset)
  {
    spec_.task = task;
    spec_.dt = kDt;
    spec_.action.contract = rl_policy::kBodyRatesContract;
    spec_.action.specific_thrust_max_ms2 = 40.0;
    spec_.action.max_body_rate_rad_s = {20.0, 20.0, 20.0};
    spec_.observation.layout = rl_policy::kGateRelativeLayout;
    spec_.observation.action_history = 2;
    spec_.observation.dim = 28;
    spec_.observation.motor_speed_normalization = 3000.0;
  }

  Action act(const Eigen::VectorXd & observation) const override
  {
    observations.push_back(observation);
    return actionAt(observations.size());
  }

  // The raw action of the call-th act(); the last entry is out of range on purpose.
  Action actionAt(std::size_t call) const
  {
    return {offset_ + 0.01 * static_cast<double>(call), -offset_, 0.5 * offset_, 2.0};
  }

  const rl_policy::PolicySpec & spec() const override {return spec_;}

  std::size_t calls() const {return observations.size();}

  mutable std::vector<Eigen::VectorXd> observations;

private:
  rl_policy::PolicySpec spec_;
  double offset_;
};

VehicleState at(double x, double y, double z, double yaw = 0.0)
{
  VehicleState state;
  state.position = Eigen::Vector3d(x, y, z);
  state.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  state.motor_speeds = {1500.0, 1500.0, 1500.0, 1500.0};
  return state;
}

VehicleState moving(double x, double y, double z, double vx, double vy, double vz)
{
  VehicleState state = at(x, y, z);
  state.velocity = Eigen::Vector3d(vx, vy, vz);
  return state;
}

std::vector<MissionEvent> ofType(const std::vector<MissionEvent> & events, MissionEvent::Type type)
{
  std::vector<MissionEvent> out;
  for (const MissionEvent & event : events) {
    if (event.type == type) {
      out.push_back(event);
    }
  }
  return out;
}

double f32(double value)
{
  return static_cast<double>(static_cast<float>(value));
}

void expectHistory(const Eigen::VectorXd & obs, const Action & newest, const Action & older)
{
  const Action clipped_newest = rl_policy::clip(newest);
  const Action clipped_older = rl_policy::clip(older);
  for (int k = 0; k < 4; ++k) {
    EXPECT_EQ(obs(20 + k), f32(clipped_newest[k])) << "a(t-1) entry " << k;
    EXPECT_EQ(obs(24 + k), f32(clipped_older[k])) << "a(t-2) entry " << k;
  }
}

void expectZeroHistory(const Eigen::VectorXd & obs)
{
  for (int i = 20; i < 28; ++i) {
    EXPECT_EQ(obs(i), 0.0) << "entry " << i;
  }
}

// Passes gates 0 and 1 and approaches gate 2 along +x; `last` is the step that crosses gate 2.
StepResult flyToTheLastGate(MissionController & mission, const VehicleState & last)
{
  const std::vector<double> xs = {4.8, 5.2, 9.8, 10.2, 14.8};
  for (std::size_t i = 0; i < xs.size(); ++i) {
    mission.step(at(xs[i], 0.0, 1.5), static_cast<double>(i) * kDt);
  }
  return mission.step(last, static_cast<double>(xs.size()) * kDt);
}

class Mission : public ::testing::Test
{
protected:
  // Three gates along +x, crossed towards +x.
  MissionConfig config(MissionType type = MissionType::Race, int laps = 1) const
  {
    MissionConfig config;
    config.type = type;
    for (int i = 0; i < 3; ++i) {
      rl_policy::Gate gate;
      gate.position = Eigen::Vector3d(5.0 * (i + 1), 0.0, 1.5);
      gate.yaw = 0.0;
      config.gates.push_back(gate);
    }
    config.bounds.x_min = -10.0;
    config.bounds.x_max = 30.0;
    config.bounds.y_min = -5.0;
    config.bounds.y_max = 5.0;
    config.laps = laps;
    config.pass_tolerance_m = 0.9;
    config.valid_half_m = 0.4;
    config.gate_timeout_s = 5.0;
    config.bounds_margin_m = 0.5;
    config.ceiling_m = 10.0;
    config.exit_debounce_steps = 3;
    config.hold_setpoint.position = Eigen::Vector3d(0.0, 0.0, 1.5);
    config.min_altitude_m = 1.0;
    config.mass_kg = 1.25;
    return config;
  }

  PolicyBank bank(bool with_race = true, bool with_hover = true) const
  {
    PolicyBank policies;
    if (with_race) {
      policies.add("race", race);
    }
    if (with_hover) {
      policies.add("hover", hover);
    }
    return policies;
  }

  std::shared_ptr<CountingPolicy> race = std::make_shared<CountingPolicy>("race", 0.1);
  std::shared_ptr<CountingPolicy> hover = std::make_shared<CountingPolicy>("hover", -0.3);
};

}  // namespace

TEST_F(Mission, RaceMissionStartsFromIdle) {
  MissionController mission(config(), bank());
  EXPECT_EQ(mission.phase(), Phase::Idle);
  EXPECT_THROW(mission.step(at(0.0, 0.0, 1.5), 0.0), std::logic_error);

  ASSERT_TRUE(mission.start());
  EXPECT_EQ(mission.phase(), Phase::Race);
  EXPECT_EQ(mission.status().target, 0);
  EXPECT_EQ(mission.status().lap, 0);
  EXPECT_EQ(mission.status().gates_passed, 0);
  EXPECT_EQ(mission.status().policy, "race");
  const auto events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].from, Phase::Idle);
  EXPECT_EQ(events[0].to, Phase::Race);
  EXPECT_TRUE(mission.takeEvents().empty());

  const StepResult first = mission.step(at(0.0, 0.0, 1.5), 0.0);
  EXPECT_EQ(race->calls(), 1u);
  EXPECT_EQ(hover->calls(), 0u);
  expectZeroHistory(first.observation);
  EXPECT_EQ(first.status.steps, 1u);

  EXPECT_FALSE(mission.start());
  EXPECT_EQ(mission.phase(), Phase::Race);
}

TEST_F(Mission, HoverMissionStartsInHoldAtTheClampedSetpoint) {
  MissionConfig hover_config = config(MissionType::Hover);
  hover_config.hold_setpoint.position = Eigen::Vector3d(40.0, -9.0, 0.2);
  hover_config.hold_setpoint.yaw = 0.7;
  MissionController mission(hover_config, bank(false, true));
  ASSERT_TRUE(mission.start());
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(mission.status().policy, "hover");
  EXPECT_EQ(mission.status().exit_reason, ExitReason::None);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(30.0, -5.0, 1.0));
  EXPECT_DOUBLE_EQ(mission.status().setpoint.yaw, 0.7);

  const StepResult result = mission.step(at(0.0, 0.0, 0.0), 0.0);
  EXPECT_EQ(hover->calls(), 1u);
  EXPECT_EQ(race->calls(), 0u);
  EXPECT_EQ(result.status.target, -1);
  EXPECT_EQ(result.frame.position, Eigen::Vector3d(30.0, -5.0, 1.0));
  for (int i = 16; i < 20; ++i) {
    EXPECT_EQ(result.observation(i), 0.0);
  }
  expectZeroHistory(result.observation);
  EXPECT_FALSE(mission.start());
}

TEST_F(Mission, ExactlyOneActPerStepAndTheTargetAdvancesFirst) {
  MissionController mission(config(), bank());
  mission.start();
  mission.step(at(4.7, 0.0, 1.5), 0.00);
  const StepResult crossing = mission.step(at(5.3, 0.1, 1.4), 0.01);
  EXPECT_EQ(race->calls(), 2u);
  EXPECT_EQ(hover->calls(), 0u);

  // The step that crosses gate 0 already observes gate 1.
  EXPECT_EQ(crossing.status.target, 1);
  EXPECT_EQ(crossing.status.gates_passed, 1);
  EXPECT_EQ(crossing.frame.position, Eigen::Vector3d(10.0, 0.0, 1.5));
  EXPECT_EQ(crossing.observation(0), f32(5.3 - 10.0));
  EXPECT_EQ(crossing.observation(2), f32(1.4 - 1.5));
  EXPECT_EQ(race->observations.back(), crossing.observation);
  EXPECT_TRUE(crossing.status.last_crossing.available);
  EXPECT_TRUE(crossing.status.last_crossing.passed);
  EXPECT_TRUE(crossing.status.last_crossing.valid);
  EXPECT_EQ(crossing.status.last_crossing.gate, 0u);

  const auto events = mission.takeEvents();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[1].type, MissionEvent::Type::GatePassed);
  EXPECT_EQ(events[1].gate, 0u);
  EXPECT_NEAR(events[1].crossing.lateral, 0.05, 1e-9);

  for (int i = 0; i < 5; ++i) {
    mission.step(at(6.0, 0.0, 1.5), 0.02 + i * kDt);
  }
  EXPECT_EQ(race->calls(), 7u);
  EXPECT_EQ(mission.status().steps, 7u);
}

TEST_F(Mission, FirstRaceStepHasNoCrossingCheck) {
  MissionController mission(config(), bank());
  mission.start();
  // Beyond gate 0's plane already: no previous position, so nothing is crossed.
  const StepResult first = mission.step(at(5.5, 0.0, 1.5), 0.0);
  EXPECT_EQ(first.status.target, 0);
  EXPECT_EQ(first.status.gates_passed, 0);
  EXPECT_EQ(mission.sequencer().stepsSincePass(), 0);
  mission.step(at(5.6, 0.0, 1.5), kDt);
  EXPECT_EQ(mission.status().target, 0);
  EXPECT_EQ(mission.sequencer().stepsSincePass(), 1);
}

TEST_F(Mission, HistoryIsZeroedOnlyWhenThePolicyChanges) {
  MissionController mission(config(), bank());
  mission.start();
  mission.step(at(0.0, 0.0, 1.5), 0.0);
  const StepResult second = mission.step(at(0.1, 0.0, 1.5), kDt);
  expectHistory(second.observation, race->actionAt(1), Action{});
  const StepResult third = mission.step(at(0.2, 0.0, 1.5), 2 * kDt);
  expectHistory(third.observation, race->actionAt(2), race->actionAt(1));
  EXPECT_EQ(third.action, rl_policy::clip(race->actionAt(3)));

  ASSERT_TRUE(mission.requestHold(1.0, 1.0, 2.0, 0.3));
  const StepResult held = mission.step(at(0.3, 0.0, 1.5), 3 * kDt);
  EXPECT_EQ(hover->calls(), 1u);
  expectZeroHistory(held.observation);
  const StepResult held_again = mission.step(at(0.3, 0.0, 1.5), 4 * kDt);
  expectHistory(held_again.observation, hover->actionAt(1), Action{});

  // A new setpoint for the same policy keeps its history.
  ASSERT_TRUE(mission.requestHold(2.0, 2.0, 2.0, 0.0));
  const StepResult retarget = mission.step(at(0.3, 0.0, 1.5), 5 * kDt);
  expectHistory(retarget.observation, hover->actionAt(2), hover->actionAt(1));
  EXPECT_EQ(retarget.frame.position, Eigen::Vector3d(2.0, 2.0, 2.0));
  EXPECT_EQ(race->calls(), 3u);
}

TEST_F(Mission, FinishHoldsWhereTheLastCrossingEnds) {
  MissionController mission(config(MissionType::Race, 1), bank());
  mission.start();
  const std::vector<double> xs = {4.8, 5.2, 9.8, 10.2, 14.8};
  for (std::size_t i = 0; i < xs.size(); ++i) {
    mission.step(at(xs[i], 0.0, 1.5), static_cast<double>(i) * kDt);
    EXPECT_EQ(mission.phase(), Phase::Race);
  }
  EXPECT_EQ(mission.status().target, 2);
  const StepResult last = mission.step(at(15.3, 0.2, 1.6, 0.2), 5 * kDt);
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(last.status.exit_reason, ExitReason::Finished);
  EXPECT_EQ(last.status.lap, 1);
  EXPECT_EQ(last.status.gates_passed, 3);
  EXPECT_EQ(last.status.target, -1);
  EXPECT_NEAR(
    (last.status.setpoint.position - Eigen::Vector3d(15.3, 0.2, 1.6)).norm(), 0.0,
    1e-12);
  EXPECT_NEAR(last.status.setpoint.yaw, 0.2, 1e-12);
  EXPECT_EQ(last.frame.position, last.status.setpoint.position);
  EXPECT_EQ(race->calls(), 5u);
  EXPECT_EQ(hover->calls(), 1u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(last.observation(i), 0.0, 1e-6);
  }
  for (int i = 16; i < 20; ++i) {
    EXPECT_EQ(last.observation(i), 0.0);
  }
  expectZeroHistory(last.observation);

  int passes = 0;
  bool finished = false;
  for (const MissionEvent & event : mission.takeEvents()) {
    passes += event.type == MissionEvent::Type::GatePassed ? 1 : 0;
    if (event.type == MissionEvent::Type::PhaseChange && event.to == Phase::Hold) {
      finished = event.reason == ExitReason::Finished && event.from == Phase::Race;
      EXPECT_FALSE(rl_policy::describe(event).empty());
    }
  }
  EXPECT_EQ(passes, 3);
  EXPECT_TRUE(finished);

  // Hold never exits.
  for (int i = 0; i < 20; ++i) {
    mission.step(at(40.0, 20.0, 20.0), (6 + i) * kDt);
  }
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(hover->calls(), 21u);
}

TEST_F(Mission, FinishWithAfterRaceHereHoldsWhereTheLastLapEnds) {
  EXPECT_EQ(MissionConfig().after_race, AfterRace::Here);
  MissionConfig here = config(MissionType::Race, 1);
  here.after_race = AfterRace::Here;
  here.hold_setpoint.position = Eigen::Vector3d(-2.0, 3.0, 2.0);
  here.hold_setpoint.yaw = 0.9;
  MissionController mission(here, bank());
  mission.start();
  const StepResult last = flyToTheLastGate(mission, at(15.3, 0.2, 1.6, 0.2));
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(last.status.exit_reason, ExitReason::Finished);
  EXPECT_NEAR(
    (last.status.setpoint.position - Eigen::Vector3d(15.3, 0.2, 1.6)).norm(), 0.0,
    1e-12);
  EXPECT_NEAR(last.status.setpoint.yaw, 0.2, 1e-12);
}

TEST_F(Mission, FinishWithAfterRaceSetpointHoldsAtTheClampedSetpoint) {
  MissionConfig to_setpoint = config(MissionType::Race, 1);
  to_setpoint.after_race = AfterRace::Setpoint;
  to_setpoint.hold_setpoint.position = Eigen::Vector3d(-2.0, 3.0, 0.4);
  to_setpoint.hold_setpoint.yaw = 0.9;
  MissionController mission(to_setpoint, bank());
  mission.start();
  const StepResult last = flyToTheLastGate(mission, at(15.3, 0.2, 1.6, 0.2));
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(last.status.exit_reason, ExitReason::Finished);
  EXPECT_EQ(last.status.lap, 1);
  EXPECT_EQ(last.status.gates_passed, 3);
  EXPECT_EQ(last.status.policy, "hover");
  const Eigen::Vector3d held(-2.0, 3.0, 1.0);
  EXPECT_EQ(last.status.setpoint.position, held);
  EXPECT_DOUBLE_EQ(last.status.setpoint.yaw, 0.9);
  EXPECT_EQ(last.frame.position, held);
  EXPECT_DOUBLE_EQ(last.frame.yaw, 0.9);

  // The first hold step already observes the setpoint, with the new policy's history zeroed.
  EXPECT_EQ(race->calls(), 5u);
  EXPECT_EQ(hover->calls(), 1u);
  EXPECT_EQ(hover->observations.back(), last.observation);
  EXPECT_NEAR(
    last.observation.head<3>().norm(), (Eigen::Vector3d(15.3, 0.2, 1.6) - held).norm(), 1e-5);
  EXPECT_NEAR(last.observation(8), 0.2 - 0.9, 1e-6);
  expectZeroHistory(last.observation);

  const auto events = mission.takeEvents();
  ASSERT_FALSE(events.empty());
  const MissionEvent & finish = events.back();
  EXPECT_EQ(finish.type, MissionEvent::Type::PhaseChange);
  EXPECT_EQ(finish.reason, ExitReason::Finished);
  EXPECT_EQ(finish.setpoint.position, held);
  EXPECT_EQ(
    rl_policy::describe(finish),
    "Race finished: 1 laps, 3 gates (race -> hold): hold at [-2.000, 3.000, 1.000] yaw 0.900 "
    "rad with policy 'hover'");
}

TEST_F(Mission, EveryOtherExitHoldsAtTheCurrentPoseWithAfterRaceSetpoint) {
  MissionConfig to_setpoint = config();
  to_setpoint.after_race = AfterRace::Setpoint;
  to_setpoint.hold_setpoint.position = Eigen::Vector3d(-2.0, 3.0, 2.0);
  to_setpoint.hold_setpoint.yaw = 0.9;

  MissionController missed(to_setpoint, bank());
  missed.start();
  missed.step(at(4.8, 1.2, 1.5), 0.0);
  missed.step(at(5.2, 1.2, 1.5, 0.1), kDt);
  EXPECT_EQ(missed.status().exit_reason, ExitReason::Missed);
  EXPECT_EQ(missed.status().setpoint.position, Eigen::Vector3d(5.2, 1.2, 1.5));
  EXPECT_NEAR(missed.status().setpoint.yaw, 0.1, 1e-12);

  MissionController timed_out(to_setpoint, bank());
  timed_out.start();
  for (int step = 0; timed_out.phase() == Phase::Race && step < 10000; ++step) {
    timed_out.step(at(4.0, 0.0, 1.5), step * kDt);
  }
  EXPECT_EQ(timed_out.status().exit_reason, ExitReason::Timeout);
  EXPECT_EQ(timed_out.status().setpoint.position, Eigen::Vector3d(4.0, 0.0, 1.5));

  MissionController outside(to_setpoint, bank());
  outside.start();
  for (int step = 0; step < to_setpoint.exit_debounce_steps; ++step) {
    outside.step(at(-10.6, 5.7, 1.5), step * kDt);
  }
  EXPECT_EQ(outside.status().exit_reason, ExitReason::OutOfBounds);
  EXPECT_EQ(outside.status().setpoint.position, Eigen::Vector3d(-10.0, 5.0, 1.5));

  MissionController requested(to_setpoint, bank());
  requested.start();
  requested.step(at(0.0, 0.0, 1.5), 0.0);
  ASSERT_TRUE(requested.requestHold(40.0, 0.5, 0.3, 1.0));
  EXPECT_EQ(requested.status().exit_reason, ExitReason::HoverRequest);
  EXPECT_EQ(requested.status().setpoint.position, Eigen::Vector3d(30.0, 0.5, 1.0));
  EXPECT_DOUBLE_EQ(requested.status().setpoint.yaw, 1.0);
}

TEST_F(Mission, AMissHoldsAtTheCurrentPose) {
  MissionController mission(config(), bank());
  mission.start();
  mission.step(at(4.8, 1.2, 1.5), 0.0);
  const StepResult miss = mission.step(at(5.2, 1.2, 1.5), kDt);
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(miss.status.exit_reason, ExitReason::Missed);
  EXPECT_EQ(miss.status.gates_passed, 0);
  EXPECT_TRUE(miss.status.last_crossing.available);
  EXPECT_FALSE(miss.status.last_crossing.passed);
  EXPECT_FALSE(miss.status.last_crossing.valid);
  EXPECT_NEAR(miss.status.last_crossing.crossing.lateral, 1.2, 1e-9);
  EXPECT_EQ(miss.status.setpoint.position, Eigen::Vector3d(5.2, 1.2, 1.5));
  EXPECT_EQ(hover->calls(), 1u);
}

TEST_F(Mission, NoPassWithinTheTimeoutHolds) {
  const MissionConfig timed = config();
  MissionController mission(timed, bank());
  mission.start();
  int updates_to_timeout = 1;
  while (!(static_cast<double>(updates_to_timeout) * kDt > timed.gate_timeout_s)) {
    ++updates_to_timeout;
  }
  // The first step makes no gate check, so the timeout lands one step later.
  const int exit_step = updates_to_timeout + 1;
  int step = 0;
  while (mission.phase() == Phase::Race && step < 10 * exit_step) {
    mission.step(at(4.0, 0.0, 1.5), step * kDt);
    ++step;
  }
  EXPECT_EQ(step, exit_step);
  EXPECT_EQ(mission.status().exit_reason, ExitReason::Timeout);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(4.0, 0.0, 1.5));
}

TEST_F(Mission, OutOfBoundsIsDebounced) {
  MissionController mission(config(), bank());
  mission.start();
  mission.step(at(0.0, 5.6, 1.5), 0.0);
  mission.step(at(0.0, 5.6, 1.5), kDt);
  mission.step(at(0.0, 0.0, 1.5), 2 * kDt);
  mission.step(at(0.0, 0.0, 10.5), 3 * kDt);
  mission.step(at(0.0, -5.6, 1.5), 4 * kDt);
  EXPECT_EQ(mission.phase(), Phase::Race);
  // Inside the margin is inside.
  mission.step(at(0.0, 5.4, 1.5), 5 * kDt);
  mission.step(at(0.0, 5.6, 1.5), 6 * kDt);
  mission.step(at(0.0, 5.6, 1.5), 7 * kDt);
  EXPECT_EQ(mission.phase(), Phase::Race);
  const StepResult exit = mission.step(at(-10.6, 5.7, 1.5), 8 * kDt);
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(exit.status.exit_reason, ExitReason::OutOfBounds);
  EXPECT_EQ(exit.status.setpoint.position, Eigen::Vector3d(-10.0, 5.0, 1.5));
}

TEST_F(Mission, HoverRequestHoldsAtTheClampedPose) {
  MissionController mission(config(), bank());
  mission.start();
  mission.step(at(0.0, 0.0, 1.5), 0.0);
  mission.takeEvents();
  EXPECT_FALSE(mission.requestHold(std::nan(""), 0.0, 1.0, 0.0));
  EXPECT_EQ(mission.phase(), Phase::Race);

  ASSERT_TRUE(mission.requestHold(40.0, 0.5, 0.3, 1.0));
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(mission.status().exit_reason, ExitReason::HoverRequest);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(30.0, 0.5, 1.0));
  EXPECT_DOUBLE_EQ(mission.status().setpoint.yaw, 1.0);
  EXPECT_EQ(mission.status().policy, "hover");
  const auto events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].from, Phase::Race);
  EXPECT_EQ(events[0].reason, ExitReason::HoverRequest);
  EXPECT_FALSE(mission.start());
}

TEST_F(Mission, HoverRequestFromIdleHolds) {
  MissionController mission(config(), bank());
  ASSERT_TRUE(mission.requestHold(1.0, 2.0, 0.0, 0.5));
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(1.0, 2.0, 1.0));
  mission.step(at(1.0, 2.0, 0.0), 0.0);
  EXPECT_EQ(hover->calls(), 1u);
  EXPECT_EQ(race->calls(), 0u);
  EXPECT_FALSE(mission.start());
}

TEST_F(Mission, CommandIsDecodedWithThePlatformMass) {
  MissionController mission(config(), bank());
  mission.start();
  const StepResult result = mission.step(at(0.0, 0.0, 1.5), 0.0);
  const Action action = rl_policy::clip(race->actionAt(1));
  EXPECT_EQ(result.action, action);
  EXPECT_DOUBLE_EQ(result.command.thrust_n, 0.5 * (action[0] + 1.0) * 40.0 * 1.25);
  EXPECT_DOUBLE_EQ(result.command.rates(0), action[1] * 20.0);
  EXPECT_DOUBLE_EQ(result.command.rates(2), 20.0);
}

TEST_F(Mission, RefusesAnIncompleteConfiguration) {
  EXPECT_THROW(MissionController(config(), bank(false, true)), std::invalid_argument);
  EXPECT_THROW(MissionController(config(), bank(true, false)), std::invalid_argument);
  EXPECT_THROW(
    MissionController(config(MissionType::Hover), bank(true, false)), std::invalid_argument);
  EXPECT_NO_THROW(MissionController(config(MissionType::Hover), bank(false, true)));

  MissionConfig bad = config();
  bad.laps = 0;
  EXPECT_THROW(MissionController(bad, bank()), std::invalid_argument);
  bad = config();
  bad.exit_debounce_steps = 0;
  EXPECT_THROW(MissionController(bad, bank()), std::invalid_argument);
  bad = config();
  bad.bounds.x_min = 40.0;
  EXPECT_THROW(MissionController(bad, bank()), std::invalid_argument);
  bad = config();
  bad.mass_kg = 0.0;
  EXPECT_THROW(MissionController(bad, bank()), std::invalid_argument);
  bad = config();
  bad.gates.clear();
  EXPECT_THROW(MissionController(bad, bank()), std::invalid_argument);
  EXPECT_THROW(rl_policy::missionTypeFromString("land"), std::invalid_argument);
  EXPECT_EQ(rl_policy::missionTypeFromString("hover"), MissionType::Hover);
}

TEST_F(Mission, OctagonWindowsReachTheSequencer) {
  MissionController square_only(config(), bank());
  const rl_policy::SequencerConfig & defaults = square_only.sequencer().config();
  EXPECT_FALSE(defaults.octagon_pass_tolerance_m.has_value());
  EXPECT_DOUBLE_EQ(defaults.passTolerance(rl_policy::GateShape::Octagon), 0.9);
  EXPECT_DOUBLE_EQ(defaults.validHalf(rl_policy::GateShape::Octagon), 0.4);

  MissionConfig octagons = config();
  octagons.gates[1].shape = rl_policy::GateShape::Octagon;
  octagons.octagon_pass_tolerance_m = 1.1;
  octagons.octagon_valid_half_m = 0.6;
  MissionController mission(octagons, bank());
  const rl_policy::SequencerConfig & given = mission.sequencer().config();
  EXPECT_DOUBLE_EQ(given.passTolerance(rl_policy::GateShape::Octagon), 1.1);
  EXPECT_DOUBLE_EQ(given.validHalf(rl_policy::GateShape::Octagon), 0.6);
  EXPECT_DOUBLE_EQ(given.passTolerance(rl_policy::GateShape::Square), 0.9);
  EXPECT_EQ(mission.sequencer().course().gate(1).shape, rl_policy::GateShape::Octagon);

  MissionConfig bad = octagons;
  bad.octagon_pass_tolerance_m = 0.0;
  EXPECT_THROW(MissionController(bad, bank()), std::invalid_argument);
  bad = octagons;
  bad.octagon_valid_half_m = -0.6;
  EXPECT_THROW(MissionController(bad, bank()), std::invalid_argument);
}

TEST_F(Mission, AfterRaceSetpointMustLieInsideTheBounds) {
  MissionConfig to_setpoint = config();
  to_setpoint.after_race = AfterRace::Setpoint;
  to_setpoint.hold_setpoint.position = Eigen::Vector3d(30.5, 0.0, 1.5);
  EXPECT_THROW(MissionController(to_setpoint, bank()), std::invalid_argument);
  to_setpoint.hold_setpoint.position = Eigen::Vector3d(0.0, -5.5, 1.5);
  EXPECT_THROW(MissionController(to_setpoint, bank()), std::invalid_argument);
  // On the bounds is inside, and z is clamped like every hold, not refused.
  to_setpoint.hold_setpoint.position = Eigen::Vector3d(30.0, -5.0, 0.0);
  EXPECT_NO_THROW(MissionController(to_setpoint, bank()));

  MissionConfig here = config();
  here.hold_setpoint.position = Eigen::Vector3d(30.5, 0.0, 1.5);
  EXPECT_NO_THROW(MissionController(here, bank()));

  // The hover mission ignores after_race.
  MissionConfig hover_config = config(MissionType::Hover);
  hover_config.after_race = AfterRace::Setpoint;
  hover_config.hold_setpoint.position = Eigen::Vector3d(40.0, -9.0, 0.2);
  MissionController mission(hover_config, bank(false, true));
  ASSERT_TRUE(mission.start());
  EXPECT_EQ(mission.status().exit_reason, ExitReason::None);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(30.0, -5.0, 1.0));

  EXPECT_EQ(rl_policy::afterRaceFromString("here"), AfterRace::Here);
  EXPECT_EQ(rl_policy::afterRaceFromString("setpoint"), AfterRace::Setpoint);
  EXPECT_THROW(rl_policy::afterRaceFromString("start"), std::invalid_argument);
  EXPECT_THROW(rl_policy::afterRaceFromString(""), std::invalid_argument);
}

TEST_F(Mission, TheHoverMissionReportsItsConfiguredSetpointBeforeItStarts) {
  MissionConfig hover_config = config(MissionType::Hover);
  hover_config.hold_setpoint.position = Eigen::Vector3d(40.0, -9.0, 0.2);
  hover_config.hold_setpoint.yaw = 0.7;
  MissionController mission(hover_config, bank(false, true));
  EXPECT_EQ(mission.phase(), Phase::Idle);
  EXPECT_FALSE(mission.status().holding);
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Configured);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(30.0, -5.0, 1.0));
  EXPECT_FALSE(mission.sentReference().has_value());
  ASSERT_TRUE(mission.start());
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Configured);
  const auto events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].source, SetpointSource::Configured);
  EXPECT_EQ(rl_policy::describe(events[0]).rfind("Holding the configured setpoint", 0), 0u);

  MissionController race(config(), bank());
  EXPECT_EQ(race.status().setpoint_source, SetpointSource::None);
}

TEST_F(Mission, ASentReferenceBeforeTheStartIsWhereTheHoldStarts) {
  MissionController mission(config(MissionType::Hover), bank(false, true));
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::Pending);
  EXPECT_EQ(mission.phase(), Phase::Idle);
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Sent);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(2.0, 1.0, 2.5));
  EXPECT_DOUBLE_EQ(mission.status().setpoint.yaw, 0.4);
  auto events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].type, MissionEvent::Type::SetpointChange);
  EXPECT_EQ(events[0].from, Phase::Idle);
  EXPECT_EQ(
    rl_policy::describe(events[0]),
    "Sent reference: the hold starts at [2.000, 1.000, 2.500] yaw 0.400 rad");

  ASSERT_TRUE(mission.start());
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(mission.status().exit_reason, ExitReason::None);
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Sent);
  events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].source, SetpointSource::Sent);
  EXPECT_EQ(rl_policy::describe(events[0]).rfind("Holding the sent reference", 0), 0u);

  const StepResult first = mission.step(at(0.0, 0.0, 0.0), 0.0);
  EXPECT_EQ(first.frame.position, Eigen::Vector3d(2.0, 1.0, 2.5));
  EXPECT_DOUBLE_EQ(first.frame.yaw, 0.4);
  expectZeroHistory(first.observation);
}

TEST_F(Mission, ASentReferenceDuringTheHoldMovesTheSetpointAndKeepsTheHistory) {
  MissionController mission(config(MissionType::Hover), bank(false, true));
  ASSERT_TRUE(mission.start());
  mission.step(at(0.0, 0.0, 1.5), 0.0);
  mission.step(at(0.0, 0.0, 1.5), kDt);
  mission.takeEvents();

  EXPECT_EQ(mission.updateHoldReference(3.0, -2.0, 2.0, -0.6), ReferenceUpdate::Moved);
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(mission.status().policy, "hover");
  EXPECT_EQ(mission.status().exit_reason, ExitReason::None);
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Sent);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(3.0, -2.0, 2.0));
  const auto events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].type, MissionEvent::Type::SetpointChange);
  EXPECT_EQ(events[0].from, Phase::Hold);
  EXPECT_EQ(
    rl_policy::describe(events[0]),
    "Sent reference: the hold moves to [3.000, -2.000, 2.000] yaw -0.600 rad");

  const StepResult moved = mission.step(at(0.0, 0.0, 1.5), 2 * kDt);
  EXPECT_EQ(moved.frame.position, Eigen::Vector3d(3.0, -2.0, 2.0));
  EXPECT_DOUBLE_EQ(moved.frame.yaw, -0.6);
  EXPECT_NEAR(moved.observation.head<3>().norm(), std::sqrt(13.25), 1e-5);
  expectHistory(moved.observation, hover->actionAt(2), hover->actionAt(1));
  EXPECT_EQ(moved.status.steps, 3u);
  EXPECT_EQ(hover->calls(), 3u);
}

TEST_F(Mission, RepublishingTheSameReferenceIsANoOp) {
  MissionController mission(config(MissionType::Hover), bank(false, true));
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::Pending);
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::Unchanged);
  mission.takeEvents();
  ASSERT_TRUE(mission.start());
  mission.step(at(0.0, 0.0, 1.5), 0.0);
  mission.step(at(0.0, 0.0, 1.5), kDt);
  mission.takeEvents();

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::Unchanged);
  }
  EXPECT_TRUE(mission.takeEvents().empty());
  EXPECT_EQ(mission.status().steps, 2u);
  EXPECT_EQ(hover->calls(), 2u);
  EXPECT_EQ(mission.history()[0], rl_policy::clip(hover->actionAt(2)));
  EXPECT_EQ(mission.history()[1], rl_policy::clip(hover->actionAt(1)));
  const StepResult next = mission.step(at(0.0, 0.0, 1.5), 2 * kDt);
  EXPECT_EQ(next.frame.position, Eigen::Vector3d(2.0, 1.0, 2.5));
  expectHistory(next.observation, hover->actionAt(2), hover->actionAt(1));
}

TEST_F(Mission, ASentReferenceIsClampedLikeTheConfiguredSetpoint) {
  MissionController mission(config(MissionType::Hover), bank(false, true));
  EXPECT_EQ(mission.updateHoldReference(40.0, -9.0, 0.2, 0.7), ReferenceUpdate::Pending);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(30.0, -5.0, 1.0));
  EXPECT_DOUBLE_EQ(mission.status().setpoint.yaw, 0.7);
  ASSERT_TRUE(mission.sentReference().has_value());
  EXPECT_EQ(mission.sentReference()->position, Eigen::Vector3d(40.0, -9.0, 0.2));

  ASSERT_TRUE(mission.start());
  mission.takeEvents();
  // Another reference that clamps to the same setpoint moves nothing.
  EXPECT_EQ(mission.updateHoldReference(50.0, -7.0, 0.5, 0.7), ReferenceUpdate::Unchanged);
  EXPECT_TRUE(mission.takeEvents().empty());
  EXPECT_EQ(mission.sentReference()->position, Eigen::Vector3d(50.0, -7.0, 0.5));
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(30.0, -5.0, 1.0));
}

TEST_F(Mission, ANonFiniteReferenceIsRefused) {
  MissionController mission(config(MissionType::Hover), bank(false, true));
  EXPECT_EQ(
    mission.updateHoldReference(std::nan(""), 1.0, 2.0, 0.0), ReferenceUpdate::NotFinite);
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(mission.updateHoldReference(1.0, 1.0, 2.0, inf), ReferenceUpdate::NotFinite);
  EXPECT_FALSE(mission.sentReference().has_value());
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Configured);
  EXPECT_TRUE(mission.takeEvents().empty());
}

TEST_F(Mission, AConfiguredHoldIgnoresSentReferences) {
  MissionConfig configured = config(MissionType::Hover);
  configured.hold_reference = HoldReference::Configured;
  MissionController mission(configured, bank(false, true));
  EXPECT_EQ(
    mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::IgnoredConfigured);
  EXPECT_FALSE(mission.sentReference().has_value());
  ASSERT_TRUE(mission.start());
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Configured);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(0.0, 0.0, 1.5));
  EXPECT_EQ(
    mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::IgnoredConfigured);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(0.0, 0.0, 1.5));
  EXPECT_EQ(MissionConfig().hold_reference, HoldReference::SentOrConfigured);
}

TEST_F(Mission, TheRaceMissionIgnoresSentReferences) {
  MissionController mission(config(), bank());
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::IgnoredRace);
  ASSERT_TRUE(mission.start());
  mission.step(at(0.0, 0.0, 1.5), 0.0);
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::IgnoredRace);
  EXPECT_EQ(mission.phase(), Phase::Race);
  ASSERT_TRUE(mission.requestHold(1.0, 1.0, 2.0, 0.3));
  mission.takeEvents();
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::IgnoredRace);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(1.0, 1.0, 2.0));
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Here);
  EXPECT_FALSE(mission.sentReference().has_value());
  EXPECT_TRUE(mission.takeEvents().empty());
}

TEST_F(Mission, AHoverRequestHoldsHereUntilTheNextSentReference) {
  MissionController mission(config(MissionType::Hover), bank(false, true));
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::Pending);
  ASSERT_TRUE(mission.start());
  ASSERT_TRUE(mission.requestHold(5.0, 0.0, 1.2, 0.1));
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Here);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(5.0, 0.0, 1.2));
  EXPECT_EQ(mission.sentReference()->position, Eigen::Vector3d(2.0, 1.0, 2.5));
  EXPECT_EQ(mission.updateHoldReference(2.0, 1.0, 2.5, 0.4), ReferenceUpdate::Moved);
  EXPECT_EQ(mission.status().setpoint_source, SetpointSource::Sent);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(2.0, 1.0, 2.5));
}

TEST_F(Mission, EveryHoldSaysWhereItsSetpointCameFrom) {
  MissionController here(config(), bank());
  here.start();
  flyToTheLastGate(here, at(15.3, 0.2, 1.6, 0.2));
  EXPECT_EQ(here.status().exit_reason, ExitReason::Finished);
  EXPECT_EQ(here.status().setpoint_source, SetpointSource::Here);

  MissionConfig to_setpoint = config();
  to_setpoint.after_race = rl_policy::AfterRace::Setpoint;
  MissionController configured(to_setpoint, bank());
  configured.start();
  flyToTheLastGate(configured, at(15.3, 0.2, 1.6, 0.2));
  EXPECT_EQ(configured.status().setpoint_source, SetpointSource::Configured);

  MissionController missed(to_setpoint, bank());
  missed.start();
  missed.step(at(4.8, 1.2, 1.5), 0.0);
  missed.step(at(5.2, 1.2, 1.5), kDt);
  EXPECT_EQ(missed.status().exit_reason, ExitReason::Missed);
  EXPECT_EQ(missed.status().setpoint_source, SetpointSource::Here);

  EXPECT_STREQ(rl_policy::toString(SetpointSource::None), "none");
  EXPECT_STREQ(rl_policy::toString(SetpointSource::Configured), "configured");
  EXPECT_STREQ(rl_policy::toString(SetpointSource::Sent), "sent");
  EXPECT_STREQ(rl_policy::toString(SetpointSource::Here), "here");
}

TEST_F(Mission, HoldReferenceIsSentOrConfiguredOrConfigured) {
  EXPECT_EQ(
    rl_policy::holdReferenceFromString("sent_or_configured"), HoldReference::SentOrConfigured);
  EXPECT_EQ(rl_policy::holdReferenceFromString("configured"), HoldReference::Configured);
  EXPECT_THROW(rl_policy::holdReferenceFromString("sent"), std::invalid_argument);
  EXPECT_THROW(rl_policy::holdReferenceFromString(""), std::invalid_argument);
  EXPECT_STREQ(rl_policy::toString(HoldReference::SentOrConfigured), "sent_or_configured");
  EXPECT_STREQ(rl_policy::toString(HoldReference::Configured), "configured");
}

TEST_F(Mission, AVirtualGateCrossedOutsideItsWindowIsNoPassAndNoMiss) {
  MissionConfig virtuals = config();
  for (rl_policy::Gate & gate : virtuals.gates) {
    gate.shape = GateShape::Virtual;
  }
  virtuals.virtual_half_m = 0.5;
  MissionController mission(virtuals, bank());
  mission.start();
  mission.takeEvents();
  mission.step(at(4.8, 0.7, 1.5), 0.0);
  const StepResult wide = mission.step(at(5.2, 0.7, 1.5), kDt);
  EXPECT_EQ(mission.phase(), Phase::Race);
  EXPECT_EQ(wide.status.target, 0);
  EXPECT_EQ(wide.status.gates_passed, 0);
  EXPECT_EQ(wide.status.exit_reason, ExitReason::None);
  EXPECT_TRUE(wide.status.last_crossing.available);
  EXPECT_FALSE(wide.status.last_crossing.passed);
  EXPECT_FALSE(wide.status.last_crossing.valid);
  EXPECT_EQ(wide.status.last_crossing.gate, 0u);
  EXPECT_NEAR(wide.status.last_crossing.crossing.lateral, 0.7, 1e-9);
  EXPECT_EQ(race->calls(), 2u);
  EXPECT_EQ(hover->calls(), 0u);
  auto events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].type, MissionEvent::Type::GateOutside);
  EXPECT_EQ(events[0].gate, 0u);
  EXPECT_DOUBLE_EQ(events[0].window_m, 0.5);
  EXPECT_EQ(
    rl_policy::describe(events[0]),
    "Gate 0 crossed outside its virtual window: lateral 0.700 m, vertical 0.000 m, not inside "
    "0.500 m; no pass, gate 0 stays the target");

  // Round again, back behind the plane, and through the window.
  mission.step(at(4.8, 0.1, 1.5), 2 * kDt);
  const StepResult pass = mission.step(at(5.2, 0.1, 1.5), 3 * kDt);
  EXPECT_EQ(mission.phase(), Phase::Race);
  EXPECT_EQ(pass.status.target, 1);
  EXPECT_EQ(pass.status.gates_passed, 1);
  EXPECT_TRUE(pass.status.last_crossing.passed);
  EXPECT_TRUE(pass.status.last_crossing.valid);
  events = mission.takeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].type, MissionEvent::Type::GatePassed);
}

TEST_F(Mission, TheVirtualFixtureSequenceFliesThroughTheMission) {
  const YAML::Node fixture = rl_policy_test::loadFixture("virtual_gates.yaml");
  MissionConfig virtuals = config(MissionType::Race, fixture["sequence"]["laps"].as<int>());
  virtuals.gates = rl_policy_test::courseGates(fixture["course"]);
  for (rl_policy::Gate & gate : virtuals.gates) {
    gate.shape = GateShape::Virtual;
  }
  virtuals.virtual_half_m = fixture["virtual_half_m"].as<double>();
  MissionController mission(virtuals, bank());
  mission.start();
  const YAML::Node positions = fixture["sequence"]["positions"];
  const YAML::Node steps = fixture["sequence"]["steps"];
  std::vector<MissionEvent> events;
  int outside = 0;
  for (std::size_t k = 0; k < positions.size(); ++k) {
    const auto p = rl_policy_test::doubles(positions[k]);
    ASSERT_EQ(mission.phase(), Phase::Race) << "position " << k;
    mission.step(at(p.at(0), p.at(1), p.at(2)), static_cast<double>(k) * kDt);
    for (const MissionEvent & event : mission.takeEvents()) {
      events.push_back(event);
    }
    if (k > 0) {
      const std::string expected = steps[k - 1]["event"].as<std::string>();
      outside += expected == "outside" ? 1 : 0;
      if (expected != "finished") {
        EXPECT_EQ(mission.status().target, steps[k - 1]["target"].as<int>()) << "step " << k;
      }
      EXPECT_EQ(mission.status().gates_passed, steps[k - 1]["gates_passed"].as<int>()) << k;
    }
  }
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(mission.status().exit_reason, ExitReason::Finished);
  EXPECT_EQ(
    ofType(events, MissionEvent::Type::GateOutside).size(), static_cast<std::size_t>(outside));
  EXPECT_EQ(ofType(events, MissionEvent::Type::GatePassed).size(), virtuals.gates.size());
  EXPECT_GE(outside, 3);
}

TEST_F(Mission, AVirtualCourseNeedsItsWindow) {
  MissionConfig virtuals = config();
  virtuals.gates[1].shape = GateShape::Virtual;
  EXPECT_THROW(MissionController(virtuals, bank()), std::invalid_argument);
  try {
    MissionController mission(virtuals, bank());
  } catch (const std::invalid_argument & e) {
    EXPECT_NE(std::string(e.what()).find("race.virtual_half_m"), std::string::npos);
  }
  EXPECT_THROW(MissionController(virtuals, bank(false, true)), std::invalid_argument);
  for (const double bad : {0.0, -0.4, std::numeric_limits<double>::quiet_NaN()}) {
    virtuals.virtual_half_m = bad;
    EXPECT_THROW(MissionController(virtuals, bank()), std::invalid_argument) << bad;
  }
  virtuals.virtual_half_m = 0.5;
  MissionController mission(virtuals, bank());
  EXPECT_DOUBLE_EQ(*mission.sequencer().config().virtual_half_m, 0.5);
  EXPECT_EQ(mission.sequencer().course().gate(1).shape, GateShape::Virtual);
}

TEST_F(Mission, ThePredictedStop) {
  const Eigen::Vector3d p(1.0, 2.0, 3.0);
  const Eigen::Vector3d v(3.0, -4.0, 0.0);
  EXPECT_TRUE(rl_policy::predictedStop(p, v, 0.1, 5.0).isApprox(
      p + 0.1 * v + v * 5.0 / 10.0, 1e-15));
  EXPECT_TRUE(rl_policy::predictedStop(p, v, 0.1, 0.0).isApprox(p + 0.1 * v, 1e-15));
  EXPECT_EQ(rl_policy::predictedStop(p, v, 0.0, 0.0), p);
  EXPECT_EQ(rl_policy::predictedStop(p, Eigen::Vector3d::Zero(), 0.2, 3.0), p);
  EXPECT_FALSE(MissionConfig().stopCheck());
}

TEST_F(Mission, TheStopCheckOffReadsThePositionAlone) {
  MissionController mission(config(), bank());
  EXPECT_FALSE(mission.config().stopCheck());
  mission.start();
  // Inside, however fast towards the edge.
  for (int step = 0; step < 10; ++step) {
    mission.step(moving(29.0, 4.0, 9.5, 80.0, 80.0, 80.0), step * kDt);
  }
  EXPECT_EQ(mission.phase(), Phase::Race);
  EXPECT_FALSE(mission.status().predicted_stop.has_value());
  for (int step = 0; step < 3; ++step) {
    mission.step(moving(31.0, 0.0, 1.5, -80.0, 0.0, 0.0), (10 + step) * kDt);
  }
  EXPECT_EQ(mission.status().exit_reason, ExitReason::OutOfBounds);
  const auto exits = ofType(mission.takeEvents(), MissionEvent::Type::PhaseChange);
  ASSERT_FALSE(exits.empty());
  EXPECT_FALSE(exits.back().predicted_stop.has_value());
  EXPECT_EQ(
    rl_policy::describe(exits.back()).rfind("Out of bounds at [31.00, 0.00, 1.50] (", 0), 0u);
}

TEST_F(Mission, APredictedStopOutsideExitsWhileThePositionIsInside) {
  MissionConfig stopping = config();
  stopping.stop_latency_s = 0.1;
  stopping.stop_decel_ms2 = 5.0;
  MissionController mission(stopping, bank());
  mission.start();
  mission.takeEvents();
  // 25 + 0.8 + 64 / 10 = 32.2, past 30 + 0.5.
  const VehicleState state = moving(25.0, 0.0, 1.5, 8.0, 0.0, 0.0);
  mission.step(state, 0.0);
  mission.step(state, kDt);
  EXPECT_EQ(mission.phase(), Phase::Race);
  ASSERT_TRUE(mission.status().predicted_stop.has_value());
  EXPECT_TRUE(mission.status().predicted_stop->isApprox(Eigen::Vector3d(32.2, 0.0, 1.5), 1e-12));
  const StepResult exit = mission.step(state, 2 * kDt);
  EXPECT_EQ(mission.phase(), Phase::Hold);
  EXPECT_EQ(exit.status.exit_reason, ExitReason::OutOfBounds);
  EXPECT_EQ(exit.status.setpoint.position, Eigen::Vector3d(25.0, 0.0, 1.5));
  EXPECT_EQ(exit.status.setpoint_source, SetpointSource::Here);
  const auto exits = ofType(mission.takeEvents(), MissionEvent::Type::PhaseChange);
  ASSERT_EQ(exits.size(), 1u);
  EXPECT_FALSE(exits[0].position_outside);
  EXPECT_TRUE(exits[0].stop_outside);
  EXPECT_EQ(
    rl_policy::describe(exits[0]),
    "Out of bounds: the predicted stop is outside, position [25.00, 0.00, 1.50], predicted stop "
    "[32.20, 0.00, 1.50] (race -> hold): hold at [25.000, 0.000, 1.500] yaw 0.000 rad with "
    "policy 'hover'");
}

TEST_F(Mission, AVelocityPointingInwardDoesNotExit) {
  MissionConfig stopping = config();
  stopping.stop_latency_s = 0.1;
  stopping.stop_decel_ms2 = 5.0;
  MissionController mission(stopping, bank());
  mission.start();
  for (int step = 0; step < 20; ++step) {
    mission.step(moving(29.0, -4.5, 9.0, -8.0, 3.0, -2.0), step * kDt);
  }
  EXPECT_EQ(mission.phase(), Phase::Race);
  EXPECT_LT(mission.status().predicted_stop->x(), 29.0);
}

TEST_F(Mission, TheCeilingIsJudgedOnTheVerticalStop) {
  MissionConfig stopping = config();
  stopping.stop_latency_s = 0.1;
  stopping.stop_decel_ms2 = 5.0;
  // 8 + 0.39 + 3.9^2 / 10 = 9.911 is below the ceiling; 8 + 0.5 + 2.5 is above it.
  MissionController below(stopping, bank());
  below.start();
  for (int step = 0; step < 10; ++step) {
    below.step(moving(0.0, 0.0, 8.0, 0.0, 0.0, 3.9), step * kDt);
  }
  EXPECT_EQ(below.phase(), Phase::Race);
  MissionController climbing(stopping, bank());
  climbing.start();
  for (int step = 0; step < 3; ++step) {
    climbing.step(moving(0.0, 0.0, 8.0, 0.0, 0.0, 5.0), step * kDt);
  }
  EXPECT_EQ(climbing.status().exit_reason, ExitReason::OutOfBounds);
  MissionController sinking(stopping, bank());
  sinking.start();
  for (int step = 0; step < 10; ++step) {
    sinking.step(moving(0.0, 0.0, 8.0, 0.0, 0.0, -5.0), step * kDt);
  }
  EXPECT_EQ(sinking.phase(), Phase::Race);
}

TEST_F(Mission, TheStopCheckIsDebouncedWithThePosition) {
  MissionConfig stopping = config();
  stopping.stop_latency_s = 0.1;
  stopping.stop_decel_ms2 = 5.0;
  MissionController mission(stopping, bank());
  mission.start();
  const VehicleState fast = moving(25.0, 0.0, 1.5, 8.0, 0.0, 0.0);
  const VehicleState slow = moving(25.0, 0.0, 1.5, 1.0, 0.0, 0.0);
  const VehicleState out = moving(31.0, 0.0, 1.5, -1.0, 0.0, 0.0);
  int step = 0;
  for (const VehicleState & state : {fast, fast, slow, fast, out, slow}) {
    mission.step(state, (step++) * kDt);
    EXPECT_EQ(mission.phase(), Phase::Race) << "step " << step;
  }
  // Two outside tests in a row, then the third step outside by either.
  mission.step(fast, (step++) * kDt);
  mission.step(out, (step++) * kDt);
  EXPECT_EQ(mission.phase(), Phase::Race);
  mission.step(fast, (step++) * kDt);
  EXPECT_EQ(mission.status().exit_reason, ExitReason::OutOfBounds);
  const auto exits = ofType(mission.takeEvents(), MissionEvent::Type::PhaseChange);
  ASSERT_FALSE(exits.empty());
  EXPECT_EQ(
    rl_policy::describe(exits.back()).rfind("Out of bounds: the predicted stop is", 0), 0u);
}

TEST_F(Mission, LatencyAloneOrDecelerationAloneSwitchesTheStopCheckOn) {
  MissionConfig latency = config();
  latency.stop_latency_s = 1.0;
  EXPECT_TRUE(latency.stopCheck());
  MissionController drifting(latency, bank());
  drifting.start();
  for (int step = 0; step < 3; ++step) {
    drifting.step(moving(25.0, 0.0, 1.5, 6.0, 0.0, 0.0), step * kDt);
  }
  EXPECT_EQ(drifting.status().exit_reason, ExitReason::OutOfBounds);
  EXPECT_TRUE(drifting.status().predicted_stop->isApprox(Eigen::Vector3d(31.0, 0.0, 1.5), 1e-12));

  MissionConfig braking = config();
  braking.stop_decel_ms2 = 5.0;
  EXPECT_TRUE(braking.stopCheck());
  MissionController mission(braking, bank());
  mission.start();
  for (int step = 0; step < 3; ++step) {
    mission.step(moving(25.0, 0.0, 1.5, 8.0, 0.0, 0.0), step * kDt);
  }
  EXPECT_EQ(mission.status().exit_reason, ExitReason::OutOfBounds);
  EXPECT_TRUE(mission.status().predicted_stop->isApprox(Eigen::Vector3d(31.4, 0.0, 1.5), 1e-12));
}

TEST_F(Mission, BothOutsideSaysBoth) {
  MissionConfig stopping = config();
  stopping.stop_latency_s = 0.1;
  MissionController mission(stopping, bank());
  mission.start();
  for (int step = 0; step < 3; ++step) {
    mission.step(moving(31.0, 0.0, 1.5, 1.0, 0.0, 0.0), step * kDt);
  }
  const auto exits = ofType(mission.takeEvents(), MissionEvent::Type::PhaseChange);
  ASSERT_FALSE(exits.empty());
  EXPECT_TRUE(exits.back().position_outside);
  EXPECT_TRUE(exits.back().stop_outside);
  EXPECT_EQ(
    rl_policy::describe(exits.back()).rfind(
      "Out of bounds: the position and the predicted stop are outside, position [31.00, 0.00, "
      "1.50], predicted stop [31.10, 0.00, 1.50] (", 0), 0u);
  EXPECT_EQ(mission.status().setpoint.position, Eigen::Vector3d(30.0, 0.0, 1.5));
}

TEST_F(Mission, TheStopCheckIsValidated) {
  for (const double bad : {-0.1, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    MissionConfig latency = config();
    latency.stop_latency_s = bad;
    EXPECT_THROW(MissionController(latency, bank()), std::invalid_argument) << bad;
    MissionConfig decel = config();
    decel.stop_decel_ms2 = bad;
    EXPECT_THROW(MissionController(decel, bank()), std::invalid_argument) << bad;
  }
}
