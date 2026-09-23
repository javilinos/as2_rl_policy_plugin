#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "as2_rl_policy/core/course.hpp"
#include "as2_rl_policy/core/gates.hpp"
#include "fixture_utils.hpp"

namespace
{

using rl_policy::Course;
using rl_policy::Gate;
using rl_policy::GateEvent;
using rl_policy::GateSequencer;

Eigen::Vector3d vector3(const YAML::Node & node)
{
  const auto values = rl_policy_test::doubles(node);
  return Eigen::Vector3d(values.at(0), values.at(1), values.at(2));
}

// A point in the gate's frame: along its normal, along its side, and up.
Eigen::Vector3d inGate(const Gate & gate, double along, double lateral, double vertical)
{
  const Eigen::Vector3d normal(std::cos(gate.yaw), std::sin(gate.yaw), 0.0);
  const Eigen::Vector3d side(-std::sin(gate.yaw), std::cos(gate.yaw), 0.0);
  return gate.position + along * normal + lateral * side + Eigen::Vector3d(0.0, 0.0, vertical);
}

rl_policy::SequencerConfig sequencerConfig(int laps)
{
  rl_policy::SequencerConfig config;
  config.laps = laps;
  config.pass_tolerance_m = 0.9;
  config.valid_half_m = 0.4;
  config.gate_timeout_s = 5.0;
  config.dt = 0.01;
  return config;
}

Course seasonTwo()
{
  return Course(
    rl_policy_test::courseGates(
      rl_policy_test::loadFixture("obs_season2.yaml")["course"]));
}

// Flies straight through the sequencer's target at the given offsets.
rl_policy::GateUpdate passTarget(GateSequencer & sequencer, double lateral, double vertical)
{
  const Gate & gate = sequencer.course().gate(sequencer.target());
  return sequencer.update(
    inGate(gate, -0.2, lateral, vertical), inGate(gate, 0.2, lateral, vertical));
}

}  // namespace

TEST(Crossing, MatchesTheFixture) {
  const YAML::Node fixture = rl_policy_test::loadFixture("crossing.yaml");
  const double tolerance = fixture["tolerance"].as<double>();
  const YAML::Node cases = fixture["cases"];
  ASSERT_GT(cases.size(), 0u);
  int crossed = 0;
  for (std::size_t c = 0; c < cases.size(); ++c) {
    const auto gate_values = rl_policy_test::doubles(cases[c]["gate"]);
    Gate gate;
    gate.position = Eigen::Vector3d(gate_values.at(0), gate_values.at(1), gate_values.at(2));
    gate.yaw = gate_values.at(3);
    const rl_policy::Crossing crossing = rl_policy::planeCrossing(
      vector3(cases[c]["prev"]), vector3(cases[c]["cur"]), gate);
    EXPECT_EQ(crossing.crossed, cases[c]["crossed"].as<bool>()) << "case " << c;
    EXPECT_NEAR(crossing.lateral, cases[c]["lateral"].as<double>(), tolerance) << "case " << c;
    EXPECT_NEAR(crossing.vertical, cases[c]["vertical"].as<double>(), tolerance) << "case " << c;
    crossed += crossing.crossed ? 1 : 0;
  }
  EXPECT_GT(crossed, 0);
}

TEST(Crossing, StrictSignRule) {
  Gate gate;
  gate.position = Eigen::Vector3d(10.0, 0.0, 1.0);
  const Eigen::Vector3d before(9.5, 0.1, 1.0);
  const Eigen::Vector3d on(10.0, 0.1, 1.0);
  const Eigen::Vector3d after(10.5, 0.1, 1.0);
  EXPECT_FALSE(rl_policy::planeCrossing(before, on, gate).crossed);
  EXPECT_FALSE(rl_policy::planeCrossing(on, after, gate).crossed);
  EXPECT_FALSE(rl_policy::planeCrossing(after, before, gate).crossed);
  const rl_policy::Crossing through = rl_policy::planeCrossing(before, after, gate);
  EXPECT_TRUE(through.crossed);
  EXPECT_NEAR(through.lateral, 0.1, 1e-12);
  EXPECT_NEAR(through.vertical, 0.0, 1e-12);
}

TEST(Sequencer, CountsLapsAndFinishes) {
  GateSequencer sequencer(seasonTwo(), sequencerConfig(2));
  const std::size_t n = sequencer.course().size();
  for (std::size_t pass = 0; pass < 2 * n; ++pass) {
    const std::size_t gate = sequencer.target();
    EXPECT_EQ(gate, pass % n);
    const rl_policy::GateUpdate update = passTarget(sequencer, 0.1, -0.1);
    EXPECT_EQ(update.gate, gate);
    EXPECT_TRUE(update.valid);
    EXPECT_EQ(sequencer.gatesPassed(), static_cast<int>(pass + 1));
    EXPECT_EQ(sequencer.lap(), static_cast<int>((pass + 1) / n));
    EXPECT_EQ(sequencer.target(), (pass + 1) % n);
    EXPECT_EQ(update.event, pass + 1 == 2 * n ? GateEvent::Finished : GateEvent::Passed);
  }
  EXPECT_TRUE(sequencer.finished());
  sequencer.reset();
  EXPECT_EQ(sequencer.target(), 0u);
  EXPECT_EQ(sequencer.lap(), 0);
  EXPECT_EQ(sequencer.gatesPassed(), 0);
  EXPECT_FALSE(sequencer.finished());
}

TEST(Sequencer, MissOutsideTheTolerance) {
  GateSequencer sequencer(seasonTwo(), sequencerConfig(1));
  const rl_policy::GateUpdate outside = passTarget(sequencer, 0.95, 0.0);
  EXPECT_EQ(outside.event, GateEvent::Missed);
  EXPECT_FALSE(outside.valid);
  EXPECT_EQ(sequencer.target(), 0u);
  EXPECT_EQ(sequencer.gatesPassed(), 0);

  GateSequencer again(seasonTwo(), sequencerConfig(1));
  const rl_policy::GateUpdate wide = passTarget(again, 0.0, 0.6);
  EXPECT_EQ(wide.event, GateEvent::Passed);
  EXPECT_FALSE(wide.valid);
  EXPECT_NEAR(wide.crossing.vertical, 0.6, 1e-9);
}

TEST(Sequencer, SharedPlaneAtTheOtherHeightIsAMiss) {
  const Course course = seasonTwo();
  ASSERT_EQ(course.size(), 12u);
  const Gate & upper = course.gate(10);
  const Gate & lower = course.gate(11);
  ASSERT_NEAR(upper.yaw, lower.yaw, 1e-12);
  ASSERT_NEAR((upper.position - lower.position).head<2>().norm(), 0.0, 1e-12);

  GateSequencer sequencer(course, sequencerConfig(1));
  for (int gate = 0; gate < 10; ++gate) {
    ASSERT_EQ(passTarget(sequencer, 0.0, 0.0).event, GateEvent::Passed);
  }
  ASSERT_EQ(sequencer.target(), 10u);
  const double drop = lower.position.z() - upper.position.z();
  const rl_policy::GateUpdate low = sequencer.update(
    inGate(upper, -0.2, 0.1, drop), inGate(upper, 0.2, 0.1, drop));
  EXPECT_TRUE(low.crossing.crossed);
  EXPECT_NEAR(low.crossing.vertical, drop, 1e-9);
  EXPECT_EQ(low.event, GateEvent::Missed);
  EXPECT_EQ(sequencer.target(), 10u);

  GateSequencer ordered(course, sequencerConfig(1));
  for (int gate = 0; gate < 11; ++gate) {
    ASSERT_EQ(passTarget(ordered, 0.0, 0.0).event, GateEvent::Passed);
  }
  EXPECT_EQ(passTarget(ordered, 0.0, 0.0).event, GateEvent::Finished);
}

TEST(Sequencer, TimesOutWithoutAPass) {
  const rl_policy::SequencerConfig config = sequencerConfig(1);
  GateSequencer sequencer(seasonTwo(), config);
  const Gate & gate = sequencer.course().gate(0);
  const Eigen::Vector3d hover = inGate(gate, -1.0, 0.0, 0.0);
  int updates = 0;
  rl_policy::GateUpdate update;
  do {
    update = sequencer.update(hover, hover);
    ++updates;
  } while (update.event == GateEvent::None && updates < 100000);
  EXPECT_EQ(update.event, GateEvent::Timeout);
  EXPECT_GT(updates * config.dt, config.gate_timeout_s);
  EXPECT_LE((updates - 1) * config.dt, config.gate_timeout_s);
  EXPECT_EQ(sequencer.stepsSincePass(), updates);
}

TEST(Sequencer, BackwardCrossingDoesNotCount) {
  GateSequencer sequencer(seasonTwo(), sequencerConfig(1));
  const Gate & gate = sequencer.course().gate(0);
  const rl_policy::GateUpdate update = sequencer.update(
    inGate(gate, 0.2, 0.0, 0.0), inGate(gate, -0.2, 0.0, 0.0));
  EXPECT_EQ(update.event, GateEvent::None);
  EXPECT_FALSE(update.crossing.crossed);
  EXPECT_EQ(sequencer.stepsSincePass(), 1);
}
