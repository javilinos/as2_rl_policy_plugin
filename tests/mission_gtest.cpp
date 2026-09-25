#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "as2_rl_policy/core/mission.hpp"

namespace
{

using rl_policy::Action;
using rl_policy::AfterRace;
using rl_policy::ExitReason;
using rl_policy::MissionConfig;
using rl_policy::MissionController;
using rl_policy::MissionEvent;
using rl_policy::MissionType;
using rl_policy::Phase;
using rl_policy::PolicyBank;
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
