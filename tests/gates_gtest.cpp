#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
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
using rl_policy::GateShape;

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

// The virtual-gate fixture's course, every gate a square, virtual or not.
std::vector<Gate> fixtureGates(const YAML::Node & fixture, bool is_virtual)
{
  std::vector<Gate> gates = rl_policy_test::courseGates(fixture["course"]);
  for (Gate & gate : gates) {
    gate.is_virtual = is_virtual;
  }
  return gates;
}

// Pass windows of `half` for squares and `octagon_half` for octagons, the valid ones tighter.
rl_policy::SequencerConfig windowConfig(int laps, double half, double octagon_half = 0.6)
{
  rl_policy::SequencerConfig config = sequencerConfig(laps);
  config.pass_tolerance_m = half;
  config.octagon_pass_tolerance_m = octagon_half;
  config.valid_half_m = 0.4;
  config.octagon_valid_half_m = 0.45;
  config.virtual_valid_half_m = 0.35;
  return config;
}

rl_policy::SequencerConfig fixtureConfig(const YAML::Node & fixture, int laps)
{
  return windowConfig(
    laps, fixture["pass_tolerance_m"].as<double>(),
    fixture["octagon_pass_tolerance_m"].as<double>());
}

GateShape shapeNamed(const std::string & name)
{
  static const std::map<std::string, GateShape> shapes = {
    {"square", GateShape::Square}, {"octagon", GateShape::Octagon}};
  return shapes.at(name);
}

GateEvent eventNamed(const std::string & name)
{
  static const std::map<std::string, GateEvent> events = {
    {"none", GateEvent::None}, {"passed", GateEvent::Passed},
    {"finished", GateEvent::Finished}, {"outside", GateEvent::Outside}};
  return events.at(name);
}

// A virtual gate of this shape at the origin facing +x, alone on its course.
Course virtualAtTheOrigin(GateShape shape = GateShape::Square)
{
  Gate gate;
  gate.shape = shape;
  gate.is_virtual = true;
  return Course(std::vector<Gate>{gate});
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

TEST(Crossing, OpeningNormOfEachShape) {
  using rl_policy::GateShape;
  EXPECT_DOUBLE_EQ(rl_policy::openingNorm(GateShape::Square, 0.3, -0.5), 0.5);
  EXPECT_DOUBLE_EQ(rl_policy::openingNorm(GateShape::Square, 0.5, 0.5), 0.5);
  // Along a flat the octagon's norm is the square's; towards a corner the diagonal edge binds.
  EXPECT_DOUBLE_EQ(rl_policy::openingNorm(GateShape::Octagon, -0.8, 0.1), 0.8);
  EXPECT_DOUBLE_EQ(rl_policy::openingNorm(GateShape::Octagon, 0.5, -0.5), 1.0 / std::sqrt(2.0));
  const double t = std::sqrt(2.0) - 1.0;
  EXPECT_NEAR(rl_policy::openingNorm(GateShape::Octagon, 0.95, 0.95 * t), 0.95, 1e-12);
  EXPECT_NEAR(rl_policy::openingNorm(GateShape::Octagon, -0.95 * t, 0.95), 0.95, 1e-12);
}

TEST(Sequencer, EverySquareIsJudgedOnItsWorseAxis) {
  const Course course = seasonTwo();
  for (int i = -12; i <= 12; ++i) {
    for (int j = -12; j <= 12; ++j) {
      // Off the grid so that no sample sits on a window's edge.
      const double lateral = 0.1 * i + 0.013;
      const double vertical = 0.1 * j + 0.013;
      const double worst = std::max(std::abs(lateral), std::abs(vertical));
      GateSequencer sequencer(course, sequencerConfig(1));
      const rl_policy::GateUpdate update = passTarget(sequencer, lateral, vertical);
      EXPECT_EQ(update.event, worst < 0.9 ? GateEvent::Passed : GateEvent::Missed)
        << lateral << ", " << vertical;
      EXPECT_EQ(update.valid, worst < 0.4) << lateral << ", " << vertical;
    }
  }
}

TEST(Sequencer, AnOctagonIsJudgedByItsOwnNormAndWindows) {
  std::vector<Gate> gates = rl_policy_test::courseGates(
    rl_policy_test::loadFixture("obs_season2.yaml")["course"]);
  gates[0].shape = rl_policy::GateShape::Octagon;
  const Course course(gates);
  rl_policy::SequencerConfig config = sequencerConfig(1);
  config.pass_tolerance_m = 1.0;
  config.valid_half_m = 0.5;
  config.octagon_pass_tolerance_m = 1.1;
  config.octagon_valid_half_m = 0.6;
  const auto through = [&](double lateral, double vertical) {
      GateSequencer sequencer(course, config);
      return passTarget(sequencer, lateral, vertical);
    };

  // Through a flat, out to the octagon's own tolerance, past the square's.
  const rl_policy::GateUpdate flat = through(1.05, 0.0);
  EXPECT_EQ(flat.event, GateEvent::Passed);
  EXPECT_FALSE(flat.valid);
  // A corner the octagon cuts off: inside the square of the same tolerance, outside the octagon.
  const rl_policy::GateUpdate corner = through(0.85, -0.85);
  EXPECT_EQ(corner.event, GateEvent::Missed);
  EXPECT_FALSE(corner.valid);
  // Beside the cut corner, inside its diagonal edge: (0.7 + 0.6) / sqrt(2) = 0.919.
  EXPECT_EQ(through(0.7, 0.6).event, GateEvent::Passed);
  // The valid window is an apothem too.
  const rl_policy::GateUpdate centred = through(0.5, 0.2);
  EXPECT_EQ(centred.event, GateEvent::Passed);
  EXPECT_TRUE(centred.valid);
  const rl_policy::GateUpdate diagonal = through(0.45, 0.45);
  EXPECT_EQ(diagonal.event, GateEvent::Passed);
  EXPECT_FALSE(diagonal.valid);

  // The gates after it are squares, held to the square's tolerance.
  GateSequencer sequencer(course, config);
  ASSERT_EQ(passTarget(sequencer, 0.0, 0.0).event, GateEvent::Passed);
  ASSERT_EQ(sequencer.target(), 1u);
  EXPECT_EQ(passTarget(sequencer, 0.85, -0.85).event, GateEvent::Passed);
  EXPECT_EQ(passTarget(sequencer, 1.05, 0.0).event, GateEvent::Missed);
}

TEST(Sequencer, AnOctagonTakesTheSquareWindowsUnlessGivenItsOwn) {
  std::vector<Gate> gates = rl_policy_test::courseGates(
    rl_policy_test::loadFixture("obs_season2.yaml")["course"]);
  gates[0].shape = rl_policy::GateShape::Octagon;
  const rl_policy::SequencerConfig config = sequencerConfig(1);
  EXPECT_DOUBLE_EQ(config.passTolerance(rl_policy::GateShape::Octagon), 0.9);
  EXPECT_DOUBLE_EQ(config.validHalf(rl_policy::GateShape::Octagon), 0.4);

  // The same windows, in the octagon's norm: 0.919 is past a 0.9 tolerance.
  GateSequencer sequencer(Course(gates), config);
  EXPECT_EQ(passTarget(sequencer, 0.7, 0.6).event, GateEvent::Missed);

  rl_policy::SequencerConfig zero = config;
  zero.octagon_pass_tolerance_m = 0.0;
  EXPECT_THROW(GateSequencer(Course(gates), zero), std::invalid_argument);
  rl_policy::SequencerConfig negative = config;
  negative.octagon_valid_half_m = -0.6;
  EXPECT_THROW(GateSequencer(Course(gates), negative), std::invalid_argument);
}

TEST(VirtualGate, EveryFixtureCrossingIsJudgedInItsShapesWindowVirtualOrNot) {
  const YAML::Node fixture = rl_policy_test::loadFixture("virtual_gates.yaml");
  const double tolerance = fixture["tolerance"].as<double>();
  const rl_policy::SequencerConfig config = fixtureConfig(fixture, 1);
  const YAML::Node cases = fixture["cases"];
  ASSERT_GT(cases.size(), 0u);
  std::map<std::string, int> outside;
  int at_the_edge = 0;
  for (std::size_t c = 0; c < cases.size(); ++c) {
    const bool crossed = cases[c]["crossed"].as<bool>();
    const bool passed = cases[c]["passed"].as<bool>();
    const GateShape shape = shapeNamed(cases[c]["shape"].as<std::string>());
    for (const bool is_virtual : {true, false}) {
      Gate gate = fixtureGates(fixture, is_virtual).at(cases[c]["gate"].as<std::size_t>());
      gate.shape = shape;
      GateSequencer sequencer(Course(std::vector<Gate>{gate}), config);
      const rl_policy::GateUpdate update =
        sequencer.update(vector3(cases[c]["prev"]), vector3(cases[c]["cur"]));
      const std::string where = "case " + std::to_string(c) + (is_virtual ? " virtual" : "");
      EXPECT_EQ(update.crossing.crossed, crossed) << where;
      EXPECT_NEAR(update.crossing.lateral, cases[c]["lateral"].as<double>(), tolerance) << where;
      EXPECT_NEAR(update.crossing.vertical, cases[c]["vertical"].as<double>(), tolerance) << where;
      // Outside the window a virtual gate lets the race go on, a physical one ends it.
      const GateEvent missed = is_virtual ? GateEvent::Outside : GateEvent::Missed;
      const GateEvent expected =
        passed ? GateEvent::Finished : (crossed ? missed : GateEvent::None);
      EXPECT_EQ(update.event, expected) << where;
      EXPECT_EQ(sequencer.gatesPassed(), passed ? 1 : 0) << where;
      EXPECT_EQ(sequencer.finished(), passed) << where;
      if (!crossed) {
        continue;
      }
      const double norm =
        rl_policy::openingNorm(shape, update.crossing.lateral, update.crossing.vertical);
      EXPECT_NEAR(norm, cases[c]["norm"].as<double>(), tolerance) << where;
      // The valid window is logged, the virtual gate's own; it never decides the pass.
      const double valid = is_virtual ? *config.virtual_valid_half_m : config.validHalf(shape);
      EXPECT_EQ(update.valid, cases[c]["norm"].as<double>() < valid) << where;
      if (is_virtual) {
        at_the_edge += cases[c]["norm"].as<double>() == config.passTolerance(shape) ? 1 : 0;
        outside[cases[c]["shape"].as<std::string>()] += passed ? 0 : 1;
      }
    }
  }
  EXPECT_GE(at_the_edge, 6);
  EXPECT_GE(outside["square"], 8);
  EXPECT_GE(outside["octagon"], 4);
}

TEST(VirtualGate, TheFixtureSequenceComesRoundAfterAnOutsideCrossing) {
  const YAML::Node fixture = rl_policy_test::loadFixture("virtual_gates.yaml");
  const double tolerance = fixture["tolerance"].as<double>();
  const YAML::Node sequence = fixture["sequence"];
  GateSequencer sequencer(
    Course(fixtureGates(fixture, true)), fixtureConfig(fixture, sequence["laps"].as<int>()));
  const YAML::Node positions = sequence["positions"];
  const YAML::Node steps = sequence["steps"];
  ASSERT_EQ(steps.size() + 1, positions.size());
  int outside = 0;
  for (std::size_t k = 0; k < steps.size(); ++k) {
    const rl_policy::GateUpdate update =
      sequencer.update(vector3(positions[k]), vector3(positions[k + 1]));
    const GateEvent expected = eventNamed(steps[k]["event"].as<std::string>());
    EXPECT_EQ(update.gate, steps[k]["gate"].as<std::size_t>()) << "step " << k;
    EXPECT_EQ(update.event, expected) << "step " << k;
    EXPECT_EQ(sequencer.target(), steps[k]["target"].as<std::size_t>()) << "step " << k;
    EXPECT_EQ(sequencer.lap(), steps[k]["lap"].as<int>()) << "step " << k;
    EXPECT_EQ(sequencer.gatesPassed(), steps[k]["gates_passed"].as<int>()) << "step " << k;
    if (update.crossing.crossed) {
      EXPECT_NEAR(update.crossing.lateral, steps[k]["lateral"].as<double>(), tolerance) << k;
      EXPECT_NEAR(update.crossing.vertical, steps[k]["vertical"].as<double>(), tolerance) << k;
    }
    outside += expected == GateEvent::Outside ? 1 : 0;
  }
  EXPECT_GE(outside, 3);
  EXPECT_TRUE(sequencer.finished());
}

TEST(VirtualGate, TheSameSequenceOnPhysicalGatesEndsAtItsFirstOutsideCrossing) {
  const YAML::Node fixture = rl_policy_test::loadFixture("virtual_gates.yaml");
  const YAML::Node sequence = fixture["sequence"];
  GateSequencer sequencer(
    Course(fixtureGates(fixture, false)), fixtureConfig(fixture, sequence["laps"].as<int>()));
  const YAML::Node positions = sequence["positions"];
  const YAML::Node steps = sequence["steps"];
  std::size_t k = 0;
  while (steps[k]["event"].as<std::string>() != "outside") {
    ASSERT_NE(
      sequencer.update(vector3(positions[k]), vector3(positions[k + 1])).event,
      GateEvent::Missed) << "step " << k;
    ++k;
  }
  const rl_policy::GateUpdate miss =
    sequencer.update(vector3(positions[k]), vector3(positions[k + 1]));
  EXPECT_EQ(miss.event, GateEvent::Missed);
  EXPECT_EQ(miss.gate, steps[k]["gate"].as<std::size_t>());
  EXPECT_EQ(sequencer.gatesPassed(), steps[k]["gates_passed"].as<int>());
}

TEST(VirtualGate, ItPassesInsideItsShapesPassWindowNotItsValidWindow) {
  // The pass window of the eight's 1.5 m gates, wider than the 0.5 m the policy trained on.
  rl_policy::SequencerConfig config = windowConfig(1, 0.75);
  config.virtual_valid_half_m = 0.5;
  const auto through = [&config](double lateral, double vertical) {
      GateSequencer sequencer(virtualAtTheOrigin(), config);
      const Gate & gate = sequencer.course().gate(0);
      return sequencer.update(
        inGate(gate, -0.2, lateral, vertical), inGate(gate, 0.2, lateral, vertical));
    };
  const rl_policy::GateUpdate centred = through(0.2, -0.1);
  EXPECT_EQ(centred.event, GateEvent::Finished);
  EXPECT_TRUE(centred.valid);
  const rl_policy::GateUpdate wide = through(0.7, 0.0);
  EXPECT_EQ(wide.event, GateEvent::Finished);
  EXPECT_FALSE(wide.valid);
  EXPECT_EQ(through(0.0, -0.74).event, GateEvent::Finished);
  EXPECT_EQ(through(0.76, 0.0).event, GateEvent::Outside);
  EXPECT_EQ(through(0.0, 0.75).event, GateEvent::Outside);
  // A physical gate of the same window misses where the virtual one lets the race go on.
  Gate physical;
  GateSequencer sequencer(Course(std::vector<Gate>{physical}), config);
  EXPECT_EQ(
    sequencer.update(inGate(physical, -0.2, 0.76, 0.0), inGate(physical, 0.2, 0.76, 0.0)).event,
    GateEvent::Missed);
}

TEST(VirtualGate, TheWindowIsStrictOnBothAxes) {
  const auto through = [](double lateral, double vertical) {
      GateSequencer sequencer(virtualAtTheOrigin(), windowConfig(1, 0.5));
      const Gate & gate = sequencer.course().gate(0);
      return sequencer.update(
        inGate(gate, -0.2, lateral, vertical), inGate(gate, 0.2, lateral, vertical)).event;
    };
  const double below = std::nextafter(0.5, 0.0);
  EXPECT_EQ(through(below, 0.0), GateEvent::Finished);
  EXPECT_EQ(through(0.5, 0.0), GateEvent::Outside);
  EXPECT_EQ(through(-0.5, 0.0), GateEvent::Outside);
  EXPECT_EQ(through(0.0, -below), GateEvent::Finished);
  EXPECT_EQ(through(0.0, 0.5), GateEvent::Outside);
  // The square's corner, which an octagon of the same apothem would cut off.
  EXPECT_EQ(through(below, -below), GateEvent::Finished);
  EXPECT_EQ(through(0.9, 0.0), GateEvent::Outside);
}

TEST(VirtualGate, AVirtualOctagonIsJudgedByTheOctagonNormAndWindow) {
  const auto through = [](double lateral, double vertical) {
      GateSequencer sequencer(virtualAtTheOrigin(GateShape::Octagon), windowConfig(1, 0.5, 0.6));
      const Gate & gate = sequencer.course().gate(0);
      return sequencer.update(
        inGate(gate, -0.2, lateral, vertical), inGate(gate, 0.2, lateral, vertical)).event;
    };
  // Along a flat, out to the octagon's window, past the square's.
  EXPECT_EQ(through(0.55, 0.0), GateEvent::Finished);
  EXPECT_EQ(through(0.0, -0.59), GateEvent::Finished);
  EXPECT_EQ(through(0.6, 0.0), GateEvent::Outside);
  // Inside the square's window, in the corner the octagon cuts off: (0.45 + 0.45) / sqrt(2).
  EXPECT_EQ(through(0.45, 0.45), GateEvent::Outside);
  EXPECT_EQ(through(-0.42, 0.42), GateEvent::Finished);
}

TEST(VirtualGate, AnOutsideCrossingNeitherAdvancesNorEndsTheRace) {
  GateSequencer sequencer(virtualAtTheOrigin(), windowConfig(2, 0.5));
  const Gate & gate = sequencer.course().gate(0);
  const rl_policy::GateUpdate wide =
    sequencer.update(inGate(gate, -0.2, 0.7, 0.0), inGate(gate, 0.2, 0.7, 0.0));
  EXPECT_EQ(wide.event, GateEvent::Outside);
  EXPECT_TRUE(wide.crossing.crossed);
  EXPECT_FALSE(wide.valid);
  EXPECT_NEAR(wide.crossing.lateral, 0.7, 1e-12);
  EXPECT_EQ(sequencer.target(), 0u);
  EXPECT_EQ(sequencer.gatesPassed(), 0);
  EXPECT_EQ(sequencer.stepsSincePass(), 1);
  EXPECT_FALSE(sequencer.finished());

  // Back behind the plane is no crossing; through the window again is the pass.
  EXPECT_EQ(
    sequencer.update(inGate(gate, 0.2, 0.7, 0.0), inGate(gate, -0.2, 0.1, 0.0)).event,
    GateEvent::None);
  const rl_policy::GateUpdate pass =
    sequencer.update(inGate(gate, -0.2, 0.1, 0.0), inGate(gate, 0.2, 0.1, 0.0));
  EXPECT_EQ(pass.event, GateEvent::Passed);
  EXPECT_TRUE(pass.valid);
  EXPECT_EQ(sequencer.gatesPassed(), 1);
  EXPECT_EQ(sequencer.lap(), 1);
  EXPECT_EQ(sequencer.stepsSincePass(), 0);
}

TEST(VirtualGate, TheTimeoutStillApplies) {
  const rl_policy::SequencerConfig config = windowConfig(1, 0.5);
  GateSequencer sequencer(virtualAtTheOrigin(), config);
  const Gate & gate = sequencer.course().gate(0);
  const Eigen::Vector3d behind = inGate(gate, -0.2, 0.7, 0.0);
  const Eigen::Vector3d beyond = inGate(gate, 0.2, 0.7, 0.0);
  int updates = 0;
  rl_policy::GateUpdate update;
  do {
    // Through the plane outside the window and back, over and over.
    const bool forward = updates % 2 == 0;
    update = sequencer.update(forward ? behind : beyond, forward ? beyond : behind);
    ++updates;
  } while ((update.event == GateEvent::None || update.event == GateEvent::Outside) &&
  updates < 100000);
  EXPECT_EQ(update.event, GateEvent::Timeout);
  EXPECT_EQ(sequencer.gatesPassed(), 0);
  EXPECT_GT(updates * config.dt, config.gate_timeout_s);
  EXPECT_LE((updates - 1) * config.dt, config.gate_timeout_s);
}

TEST(VirtualGate, PhysicalAndVirtualGatesShareTheirShapesWindows) {
  std::vector<Gate> gates(4);
  for (int i = 0; i < 4; ++i) {
    gates[i].position = Eigen::Vector3d(5.0 * (i + 1), 0.0, 1.5);
  }
  gates[1].is_virtual = true;
  gates[2].shape = GateShape::Octagon;
  gates[2].is_virtual = true;
  gates[3].shape = GateShape::Octagon;
  const rl_policy::SequencerConfig config = windowConfig(1, 0.5, 0.6);
  EXPECT_DOUBLE_EQ(config.passTolerance(GateShape::Square), 0.5);
  EXPECT_DOUBLE_EQ(config.passTolerance(GateShape::Octagon), 0.6);
  EXPECT_DOUBLE_EQ(config.validHalf(gates[0]), 0.4);
  EXPECT_DOUBLE_EQ(config.validHalf(gates[1]), 0.35);
  EXPECT_DOUBLE_EQ(config.validHalf(gates[2]), 0.35);
  EXPECT_DOUBLE_EQ(config.validHalf(gates[3]), 0.45);
  EXPECT_EQ(rl_policy::countVirtual(Course(gates)), 2u);
  EXPECT_EQ(rl_policy::countShape(Course(gates), GateShape::Octagon), 2u);

  GateSequencer missed(Course(gates), config);
  EXPECT_EQ(passTarget(missed, 0.55, 0.0).event, GateEvent::Missed);

  GateSequencer sequencer(Course(gates), config);
  EXPECT_EQ(passTarget(sequencer, 0.45, 0.0).event, GateEvent::Passed);
  EXPECT_EQ(passTarget(sequencer, 0.55, 0.0).event, GateEvent::Outside);
  EXPECT_EQ(sequencer.target(), 1u);
  const rl_policy::GateUpdate virtual_square = passTarget(sequencer, 0.45, -0.3);
  EXPECT_EQ(virtual_square.event, GateEvent::Passed);
  EXPECT_FALSE(virtual_square.valid);
  EXPECT_EQ(passTarget(sequencer, 0.45, 0.45).event, GateEvent::Outside);
  EXPECT_EQ(sequencer.target(), 2u);
  EXPECT_EQ(passTarget(sequencer, 0.55, 0.0).event, GateEvent::Passed);
  EXPECT_EQ(passTarget(sequencer, 0.45, 0.45).event, GateEvent::Missed);
  EXPECT_EQ(sequencer.target(), 3u);
}

TEST(VirtualGate, ItsValidWindowIsItsShapesUnlessGivenAndMustBeAWindow) {
  const Course virtuals = virtualAtTheOrigin();
  GateSequencer square_only(seasonTwo(), sequencerConfig(1));
  EXPECT_FALSE(square_only.config().virtual_valid_half_m.has_value());
  EXPECT_EQ(rl_policy::countVirtual(square_only.course()), 0u);
  // No key a virtual course needs beyond the physical one's.
  GateSequencer plain(virtuals, sequencerConfig(1));
  EXPECT_DOUBLE_EQ(plain.config().validHalf(virtuals.gate(0)), 0.4);
  EXPECT_DOUBLE_EQ(plain.config().passTolerance(virtuals.gate(0).shape), 0.9);
  for (const double bad : {0.0, -0.5, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    rl_policy::SequencerConfig config = sequencerConfig(1);
    config.virtual_valid_half_m = bad;
    EXPECT_THROW(GateSequencer(virtuals, config), std::invalid_argument) << bad;
    EXPECT_THROW(GateSequencer(seasonTwo(), config), std::invalid_argument) << bad;
  }
  EXPECT_NO_THROW(GateSequencer(seasonTwo(), windowConfig(1, 0.5)));
  EXPECT_STREQ(rl_policy::toString(GateShape::Square), "square");
  EXPECT_STREQ(rl_policy::toString(GateShape::Octagon), "octagon");
}
