#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "as2_rl_policy/core/mission.hpp"
#include "as2_rl_policy/core/takeover.hpp"
#include "fixture_utils.hpp"

namespace
{

using rl_policy::Action;
using rl_policy::BarKind;
using rl_policy::BarReadings;
using rl_policy::BarSide;
using rl_policy::Course;
using rl_policy::MissionConfig;
using rl_policy::MissionController;
using rl_policy::MissionType;
using rl_policy::MlpPolicy;
using rl_policy::Phase;
using rl_policy::PolicyBank;
using rl_policy::PolicyError;
using rl_policy::StepResult;
using rl_policy::Takeover;
using rl_policy::TakeoverMode;
using rl_policy::TakeoverNetworks;
using rl_policy::VehicleState;

constexpr double kDt = 0.01;
constexpr double kExact = 1.0e-12;
constexpr std::size_t kGate07 = 7;
constexpr std::size_t kGate12 = 15;

// quadlab.cli.gen_takeover_fixtures' columns of a tick.
constexpr std::size_t kTarget = 1;
constexpr std::size_t kPosition = 2;
constexpr std::size_t kVelocity = 5;
constexpr std::size_t kMode = 8;
constexpr std::size_t kObstacle = 9;
constexpr std::size_t kGain = 10;
constexpr std::size_t kReading = 11;
constexpr std::size_t kValues = 12;
constexpr std::size_t kRow = 14;
constexpr std::size_t kAction = 45;
constexpr std::size_t kMain = 49;
constexpr std::size_t kAim = 53;
constexpr std::size_t kAimed = 55;
constexpr std::size_t kResidual = 59;
constexpr std::size_t kRowTail = 63;

std::shared_ptr<const TakeoverNetworks> networks()
{
  static const auto shared =
    TakeoverNetworks::load(rl_policy_test::fixturePath("takeover.yaml"));
  return shared;
}

// A recurrent aim (format 3), and a residual that reads what it applies.
std::shared_ptr<const TakeoverNetworks> memoryNetworks()
{
  static const auto shared =
    TakeoverNetworks::load(rl_policy_test::fixturePath("takeover_memory.yaml"));
  return shared;
}

std::shared_ptr<MlpPolicy> mainPolicy()
{
  static const auto shared = MlpPolicy::load(rl_policy_test::fixturePath("takeover_main.yaml"));
  return shared;
}

// The race policy's act(), counted.
class CountingMain : public rl_policy::Policy
{
public:
  Action act(const Eigen::VectorXd & observation) const override
  {
    ++calls;
    return mainPolicy()->act(observation);
  }

  const rl_policy::PolicySpec & spec() const override {return mainPolicy()->spec();}

  mutable std::size_t calls = 0;
};

// A policy that is not the one the networks trained around.
class OtherPolicy : public rl_policy::Policy
{
public:
  explicit OtherPolicy(rl_policy::PolicySpec spec)
  : spec_(std::move(spec)) {}

  Action act(const Eigen::VectorXd &) const override {return Action{};}

  const rl_policy::PolicySpec & spec() const override {return spec_;}

private:
  rl_policy::PolicySpec spec_;
};

std::vector<rl_policy::Gate> trainedGates()
{
  return mainPolicy()->spec().course->gates;
}

VehicleState at(const Eigen::Vector3d & position, const Eigen::Vector3d & velocity)
{
  VehicleState state;
  state.position = position;
  state.velocity = velocity;
  state.motor_speeds = {1500.0, 1500.0, 1500.0, 1500.0};
  return state;
}

// On the gate's normal, `along` m past its plane, at 5 m/s through it.
VehicleState nearGate(const rl_policy::Gate & gate, double along)
{
  const Eigen::Vector3d normal(std::cos(gate.yaw), std::sin(gate.yaw), 0.0);
  return at(gate.position + along * normal, 5.0 * normal);
}

std::unique_ptr<Takeover> takeover(BarSide one_at = BarSide::Right)
{
  return std::make_unique<Takeover>(
    networks(), std::vector<BarKind>{BarKind::UpDown, BarKind::Rotating}, one_at);
}

Eigen::VectorXd zeroRow()
{
  return Eigen::VectorXd::Zero(rl_policy::kMainRowSize);
}

class TakeoverMission : public ::testing::Test
{
protected:
  MissionConfig config(MissionType type = MissionType::Race) const
  {
    MissionConfig config;
    config.type = type;
    config.gates = trainedGates();
    config.bounds.x_min = -5.0;
    config.bounds.x_max = 40.0;
    config.bounds.y_min = -5.0;
    config.bounds.y_max = 45.0;
    config.laps = 3;
    config.pass_tolerance_m = 0.9;
    config.valid_half_m = 0.5;
    config.gate_timeout_s = 50.0;
    config.bounds_margin_m = 0.0;
    config.ceiling_m = 10.0;
    config.exit_debounce_steps = 3;
    config.hold_setpoint.position = Eigen::Vector3d(5.0, 4.0, 1.5);
    config.min_altitude_m = 1.0;
    config.mass_kg = 1.0;
    return config;
  }

  PolicyBank bank() const
  {
    PolicyBank policies;
    policies.add("race", race);
    policies.add("hover", hover);
    return policies;
  }

  // Through gates 0 .. count - 1 in order, each on its normal.
  void passGates(MissionController & mission, std::size_t count)
  {
    for (std::size_t i = 0; i < count; ++i) {
      for (const double along : {-0.2, 0.2}) {
        mission.step(nearGate(mission.config().gates[i], along), t_);
        t_ += kDt;
      }
    }
  }

  StepResult stepAt(MissionController & mission, const VehicleState & state)
  {
    const StepResult result = mission.step(state, t_);
    t_ += kDt;
    return result;
  }

  std::shared_ptr<CountingMain> race = std::make_shared<CountingMain>();
  std::shared_ptr<CountingMain> hover = std::make_shared<CountingMain>();
  double t_ = 100.0;
};

}  // namespace

TEST(BarValues, AreTheKernelsScale) {
  const YAML::Node table = rl_policy_test::loadFixture("takeover_readings.yaml")["readings"];
  ASSERT_GT(table.size(), 4u);
  for (const YAML::Node & row : table) {
    const double position = row["position"].as<double>();
    const auto check = [position](BarKind kind, BarSide side, const YAML::Node & expected) {
        const std::array<double, 2> values = rl_policy::barValues(kind, position, side);
        EXPECT_NEAR(values[0], expected[0].as<double>(), kExact) << "position " << position;
        EXPECT_NEAR(values[1], expected[1].as<double>(), kExact) << "position " << position;
      };
    check(BarKind::UpDown, BarSide::Left, row["vertical"]);
    check(BarKind::UpDown, BarSide::Right, row["vertical"]);
    check(BarKind::LeftRight, BarSide::Left, row["horizontal_left"]);
    check(BarKind::LeftRight, BarSide::Right, row["horizontal_right"]);
    check(BarKind::Rotating, BarSide::Left, row["rotating"]);
    check(BarKind::Rotating, BarSide::Right, row["rotating"]);
  }
}

TEST(BarValues, ALeftRightBarAtTheLeftIsOnTheKernelsMinusSide) {
  // The kernel's plus is the approach's right; obstacle_gates' 1 is the gate's left.
  EXPECT_EQ(rl_policy::barValues(BarKind::LeftRight, 1.0, BarSide::Left)[0], -1.0);
  EXPECT_EQ(rl_policy::barValues(BarKind::LeftRight, 0.0, BarSide::Left)[0], 1.0);
  EXPECT_EQ(rl_policy::barValues(BarKind::LeftRight, 1.0, BarSide::Right)[0], 1.0);
  EXPECT_EQ(rl_policy::barValues(BarKind::UpDown, 1.0, BarSide::Left)[0], 1.0);
}

TEST(BarValues, TakeTheMissionsWords) {
  EXPECT_EQ(rl_policy::barKindFromString("vertical"), BarKind::UpDown);
  EXPECT_EQ(rl_policy::barKindFromString("horizontal"), BarKind::LeftRight);
  EXPECT_EQ(rl_policy::barKindFromString("rotating"), BarKind::Rotating);
  EXPECT_THROW(rl_policy::barKindFromString("updown"), std::invalid_argument);
  EXPECT_EQ(rl_policy::barSideFromString("left"), BarSide::Left);
  EXPECT_EQ(rl_policy::barSideFromString("right"), BarSide::Right);
  EXPECT_THROW(rl_policy::barSideFromString("up"), std::invalid_argument);
}

TEST(TakeoverFile, CarriesWhatDefinesItsNetworks) {
  const rl_policy::TakeoverSpec & spec = networks()->spec();
  EXPECT_EQ(spec.width(), 70);
  EXPECT_EQ(spec.history, 10);
  EXPECT_EQ(spec.dt, 0.01);
  EXPECT_EQ(spec.aim_m, 0.8);
  EXPECT_EQ(spec.authority, 2.0);
  EXPECT_EQ(spec.handback_m, 3.0);
  EXPECT_EQ(spec.reading_hz, 18.0);
  EXPECT_EQ(spec.main_sha256, mainPolicy()->spec().source.checkpoint_sha256);
  ASSERT_EQ(spec.gates.size(), 2u);
  EXPECT_EQ(spec.gates[0].name, "gate07");
  EXPECT_EQ(spec.gates[0].index, kGate07);
  EXPECT_EQ(spec.gates[0].open_s, 0.0);
  EXPECT_EQ(spec.gates[0].kinds, (std::vector<BarKind>{BarKind::UpDown, BarKind::LeftRight}));
  EXPECT_EQ(spec.gates[1].name, "gate12_oct");
  EXPECT_EQ(spec.gates[1].index, kGate12);
  EXPECT_EQ(spec.gates[1].open_s, 0.9);
  EXPECT_EQ(spec.gates[1].kinds, (std::vector<BarKind>{BarKind::Rotating}));
  EXPECT_GE(networks()->goldenCheck().rows, 64u);
  EXPECT_LE(networks()->goldenCheck().max_error, networks()->goldenCheck().tolerance);
}

TEST(TakeoverFile, IsRefusedWhenItIsNotOne) {
  EXPECT_THROW(TakeoverNetworks::load("/no/such/file.yaml"), PolicyError);
  // A race policy is a file of another format.
  EXPECT_THROW(
    TakeoverNetworks::load(rl_policy_test::fixturePath("takeover_main.yaml")), PolicyError);
}

TEST(TakeoverFile, IsRefusedWhenItsWeightsMissItsGoldens) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover.yaml");
  YAML::Node bias = doc["residual"]["layers"][3]["bias"];
  bias[0] = bias[0].as<double>() + 0.5;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, WithoutFramesReadsOneFramePerReading) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover.yaml");
  ASSERT_TRUE(doc["reading"]["frame_hz"].IsDefined());
  EXPECT_EQ(networks()->spec().frame_hz, 80.0);
  doc["reading"].remove("frame_hz");
  const auto older = TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover"));
  EXPECT_EQ(older->spec().frame_hz, 0.0);
  doc["reading"]["frame_hz"] = -1.0;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, IsReadFromItsFirstVersionToItsLast) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover.yaml");
  // Neither a recurrent aim nor show_aim: written at 2, for the plugins that fly today.
  EXPECT_EQ(doc["format_version"].as<int>(), 2);
  doc["format_version"] = rl_policy::kTakeoverFormatVersion;
  EXPECT_NO_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")));
  doc["format_version"] = 1;
  EXPECT_NO_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")));
  doc["format_version"] = 0;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
  doc["format_version"] = rl_policy::kTakeoverFormatVersion + 1;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, WithARecurrentAimCarriesItsMemoryAndTheResidualsTwoColumns) {
  const rl_policy::TakeoverSpec & spec = memoryNetworks()->spec();
  EXPECT_TRUE(memoryNetworks()->recurrent());
  EXPECT_EQ(memoryNetworks()->memorySize(), 128);
  EXPECT_TRUE(spec.show_aim);
  EXPECT_EQ(spec.width(), 70);
  EXPECT_EQ(spec.residualWidth(), 72);
  EXPECT_EQ(spec.frame_hz, 83.0);
  EXPECT_EQ(spec.aim_m, 1.0);
  EXPECT_EQ(spec.main_sha256, mainPolicy()->spec().source.checkpoint_sha256);
  EXPECT_GE(memoryNetworks()->goldenCheck().rows, 64u);
  EXPECT_LE(memoryNetworks()->goldenCheck().max_error, memoryNetworks()->goldenCheck().tolerance);
  EXPECT_FALSE(networks()->recurrent());
  EXPECT_EQ(networks()->memorySize(), 0);
  EXPECT_FALSE(networks()->spec().show_aim);
  EXPECT_EQ(networks()->spec().residualWidth(), 70);
}

TEST(TakeoverFile, HoldsARecurrentAimAndShowAimFromItsThirdVersionOn) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  EXPECT_EQ(doc["format_version"].as<int>(), 3);
  doc["format_version"] = 2;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, HoldsAHandBackPastTheNextGateFromItsFifthVersionOn) {
  EXPECT_LT(networks()->spec().handback_next_m, 0.0);
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  doc["format_version"] = 5;
  doc["window"]["handback_next_m"] = 1.0;
  const auto later = TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover"));
  EXPECT_EQ(later->spec().handback_next_m, 1.0);
  EXPECT_EQ(later->spec().handback_m, 3.0);
  doc["window"]["handback_next_m"] = -0.5;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
  doc["window"]["handback_next_m"] = 1.0;
  doc["format_version"] = 3;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, HoldsTheRotatingBarsLaneFromItsSixthVersionOn) {
  EXPECT_EQ(networks()->spec().lane_hold_s, 0.0);
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  doc["format_version"] = 6;
  doc["window"]["lane_hold_s"] = 0.45;
  const auto later = TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover"));
  EXPECT_EQ(later->spec().lane_hold_s, 0.45);
  doc["window"]["lane_hold_s"] = -0.1;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
  doc["window"]["lane_hold_s"] = 0.45;
  doc["format_version"] = 5;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, IsRefusedWithAnAimOnARule) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  doc["format_version"] = 5;
  doc["aim"]["rule"]["kind"] = "fit";
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, IsRefusedWhenItsRecurrentAimReadsPastTheRow) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  doc["aim"]["memory"]["columns"][0] = 70;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, IsRefusedWhenItsAimsMemoryMissesItsGoldens) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  YAML::Node bias = doc["aim"]["layers"][2]["bias"];
  bias[0] = bias[0].as<double>() + 0.5;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, IsRefusedWhenTheResidualHasNoRowsOfItsOwn) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  doc["golden"].remove("residual_input");
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(TakeoverFile, IsRefusedWithAnotherHistoryThanItsNetworksRead) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover.yaml");
  doc["reading"]["history"] = 8;
  EXPECT_THROW(TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")), PolicyError);
}

TEST(BarKinds, FollowTheFilesGatesInAnyOrderNamed) {
  EXPECT_EQ(
    rl_policy::barKinds(networks()->spec(), {"gate12_oct", "gate07"}, {"rotating", "horizontal"}),
    (std::vector<BarKind>{BarKind::LeftRight, BarKind::Rotating}));
}

TEST(BarKinds, TakeNoneForAGateWithoutItsBar) {
  const rl_policy::TakeoverSpec & spec = networks()->spec();
  EXPECT_EQ(
    rl_policy::barKinds(spec, {"gate07", "gate12_oct"}, {"horizontal", "none"}),
    (std::vector<BarKind>{BarKind::LeftRight, BarKind::None}));
  // Said, not forgotten: a gate left out is still refused.
  EXPECT_THROW(rl_policy::barKinds(spec, {"gate07"}, {"horizontal"}), std::invalid_argument);
  EXPECT_THROW(
    Takeover(networks(), std::vector<BarKind>{BarKind::None, BarKind::None}, BarSide::Left),
    std::invalid_argument);
}

TEST(BarKinds, RefuseWhatTheNetworksDidNotTrainOn) {
  const rl_policy::TakeoverSpec & spec = networks()->spec();
  EXPECT_THROW(
    rl_policy::barKinds(spec, {"gate07", "gate12_oct"}, {"rotating", "rotating"}),
    std::invalid_argument);
  EXPECT_THROW(
    rl_policy::barKinds(spec, {"gate07", "gate12_oct"}, {"vertical", "vertical"}),
    std::invalid_argument);
  EXPECT_THROW(rl_policy::barKinds(spec, {"gate07"}, {"vertical"}), std::invalid_argument);
  EXPECT_THROW(
    rl_policy::barKinds(spec, {"gate07", "gate12_oct"}, {"vertical"}), std::invalid_argument);
  EXPECT_THROW(
    rl_policy::barKinds(
      spec, {"gate07", "gate12_oct", "gate01"}, {"vertical", "rotating", "vertical"}),
    std::invalid_argument);
}

TEST(Readings, HoldTheNewestFirstWithTheirAges) {
  BarReadings readings(3, 18.0, 1.0);
  readings.keep(1.0, {0.5, 0.0});
  readings.keep(1.1, {-0.25, 0.5});
  Eigen::VectorXd out = Eigen::VectorXd::Constant(9, 7.0);
  readings.write(1.2, out);
  EXPECT_DOUBLE_EQ(out(0), -0.25);
  EXPECT_DOUBLE_EQ(out(1), 0.5);
  EXPECT_NEAR(out(2), 0.1, kExact);
  EXPECT_DOUBLE_EQ(out(3), 0.5);
  EXPECT_DOUBLE_EQ(out(4), 0.0);
  EXPECT_NEAR(out(5), 0.2, kExact);
  // Nothing read: zeros, and the longest age.
  EXPECT_EQ(out(6), 0.0);
  EXPECT_EQ(out(7), 0.0);
  EXPECT_EQ(out(8), 1.0);
}

TEST(Readings, AreGoneAtTheirMaximumAge) {
  BarReadings readings(2, 18.0, 1.0);
  readings.keep(1.0, {0.5, 0.25});
  Eigen::VectorXd out(6);
  readings.write(1.999, out);
  EXPECT_EQ(out(0), 0.5);
  readings.write(2.0, out);
  EXPECT_EQ(out(0), 0.0);
  EXPECT_EQ(out(1), 0.0);
  EXPECT_EQ(out(2), 1.0);
}

TEST(Readings, NeverAgeBelowZero) {
  BarReadings readings(1, 18.0, 1.0);
  readings.keep(1.0, {0.5, 0.25});
  Eigen::VectorXd out(3);
  readings.write(0.9, out);
  EXPECT_EQ(out(0), 0.5);
  EXPECT_EQ(out(2), 0.0);
}

TEST(Readings, AtEightyHertzAreThinnedToTheRateTheNetworksTrainedOn) {
  BarReadings readings(10, 18.0, 1.0);
  std::vector<double> kept;
  for (int frame = 0; frame < 160; ++frame) {
    const double stamp = 5.0 + frame / 80.0;
    if (readings.add(stamp, {0.0, 0.0})) {
      kept.push_back(stamp);
    }
  }
  EXPECT_NEAR(static_cast<double>(kept.size()), 36.0, 1.0);
  for (std::size_t i = 1; i < kept.size(); ++i) {
    // Whole frames, the 5 or 6 steps of 10 ms the networks saw in training or just above.
    EXPECT_GE(kept[i] - kept[i - 1], 0.05 - 1e-9);
    EXPECT_LE(kept[i] - kept[i - 1], 0.0625 + 1e-9);
  }
  // The history spans what it spanned in training, half a second, not an eighth.
  Eigen::VectorXd out(30);
  readings.write(kept.back(), out);
  EXPECT_GT(out(29), 0.45);
}

TEST(Readings, WithLatestTheNewestIsEveryFrameAndTheHistoryKeepsItsRate) {
  BarReadings readings(10, 18.0, 1.0, true);
  for (int frame = 0; frame < 160; ++frame) {
    const double stamp = 5.0 + frame / 80.0;
    EXPECT_TRUE(readings.add(stamp, {static_cast<double>(frame), 0.0}));
    EXPECT_EQ(readings.newest(), stamp);
  }
  Eigen::VectorXd out(30);
  readings.write(5.0 + 159.0 / 80.0, out);
  EXPECT_EQ(out(0), 159.0);
  EXPECT_EQ(out(2), 0.0);
  for (int slot = 2; slot < 10; ++slot) {
    const double step = out(3 * slot + 2) - out(3 * (slot - 1) + 2);
    EXPECT_GE(step, 0.05 - 1e-9) << "slot " << slot;
    EXPECT_LE(step, 0.0625 + 1e-9) << "slot " << slot;
  }
}

TEST(Readings, WithLatestStillRefuseALateOrARepeatedFrame) {
  BarReadings readings(10, 18.0, 1.0, true);
  EXPECT_TRUE(readings.add(2.0, {0.5, 0.0}));
  EXPECT_FALSE(readings.add(2.0, {0.25, 0.0}));
  EXPECT_FALSE(readings.add(1.5, {0.25, 0.0}));
  Eigen::VectorXd out(30);
  readings.write(2.0, out);
  EXPECT_EQ(out(0), 0.5);
}

TEST(Readings, AtTheTrainedRateAreAllKept) {
  BarReadings readings(10, 18.0, 1.0);
  int kept = 0;
  for (int frame = 0; frame < 18; ++frame) {
    kept += readings.add(1.0 + frame / 18.0, {0.0, 0.0});
  }
  EXPECT_EQ(kept, 18);
}

TEST(Readings, RefuseALateOrARepeatedFrame) {
  BarReadings readings(10, 18.0, 1.0);
  EXPECT_TRUE(readings.add(2.0, {0.5, 0.0}));
  EXPECT_FALSE(readings.add(2.0, {0.5, 0.0}));
  EXPECT_FALSE(readings.add(1.5, {0.5, 0.0}));
  EXPECT_EQ(readings.newest(), 2.0);
}

TEST(Readings, TakeTheFirstFrameAfterAGapAndThinFromIt) {
  BarReadings readings(10, 18.0, 1.0);
  EXPECT_TRUE(readings.add(1.0, {0.0, 0.0}));
  EXPECT_TRUE(readings.add(3.0, {0.0, 0.0}));
  EXPECT_FALSE(readings.add(3.0125, {0.0, 0.0}));
  EXPECT_TRUE(readings.add(3.0625, {0.0, 0.0}));
}

TEST(Readings, ClearToNothingRead) {
  BarReadings readings(2, 18.0, 1.0);
  readings.add(1.0, {0.5, 0.0});
  readings.clear();
  EXPECT_EQ(readings.newest(), -std::numeric_limits<double>::infinity());
  EXPECT_TRUE(readings.add(1.01, {0.5, 0.0}));
}

TEST(TakeoverTick, TakesOnlyTheNextObstaclesReadings) {
  auto pilot = takeover();
  pilot->start(0, trainedGates().size());
  EXPECT_EQ(pilot->obstacle(), 0u);
  EXPECT_FALSE(pilot->addReading("gate12_oct", 0.5, 1.0));
  EXPECT_FALSE(pilot->addReading("gate01", 0.5, 1.0));
  EXPECT_TRUE(pilot->addReading("gate07", 0.5, 1.0));
  EXPECT_EQ(pilot->readingsSeen(), 3u);
  EXPECT_EQ(pilot->readingsKept(), 1u);
}

TEST(TakeoverTick, RefusesAPositionOutsideItsRange) {
  auto pilot = takeover();
  pilot->start(0, trainedGates().size());
  EXPECT_FALSE(pilot->addReading("gate07", 1.5, 1.0));
  EXPECT_FALSE(pilot->addReading("gate07", -0.1, 1.0));
  EXPECT_FALSE(pilot->addReading("gate07", std::nan(""), 1.0));
  EXPECT_FALSE(pilot->addReading("gate07", 0.5, std::nan("")));
  EXPECT_TRUE(pilot->addReading("gate07", 1.0, 1.0));
}

TEST(TakeoverTick, StartsAtTheFirstBarredGateAtOrAfterTheTarget) {
  const std::size_t gates = trainedGates().size();
  auto pilot = takeover();
  pilot->start(kGate07, gates);
  EXPECT_EQ(pilot->obstacle(), 0u);
  pilot->start(kGate07 + 1, gates);
  EXPECT_EQ(pilot->obstacle(), 1u);
  pilot->start(kGate12, gates);
  EXPECT_EQ(pilot->obstacle(), 1u);
}

TEST(TakeoverTick, PutsThePositionInTheRowOnTheKernelsScale) {
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  const auto main = mainPolicy();
  for (const BarSide one_at : {BarSide::Left, BarSide::Right}) {
    Takeover pilot(
      networks(), std::vector<BarKind>{BarKind::LeftRight, BarKind::Rotating}, one_at);
    pilot.start(kGate07, course.size());
    const VehicleState state = nearGate(gates[kGate07], -2.0);
    ASSERT_TRUE(pilot.addReading("gate07", 0.75, 9.97));
    pilot.advance(state, course, kGate07);
    pilot.advance(state, course, kGate07);
    ASSERT_GT(pilot.gain(), 0.0);
    pilot.act(*main, zeroRow(), state, course, 10.0);
    const Eigen::VectorXd & row = pilot.tick().row;
    ASSERT_EQ(row.size(), 70);
    // The sliding bar's value, then its zero and its age; the bar's type after the readings.
    EXPECT_FLOAT_EQ(row(35), one_at == BarSide::Left ? -0.5 : 0.5);
    EXPECT_EQ(row(36), 0.0);
    EXPECT_NEAR(row(37), 0.03, 1e-6);
    EXPECT_EQ(row(65), 0.0);
    EXPECT_EQ(row(66), 1.0);
    EXPECT_EQ(row(67), 0.0);
  }
}

TEST(TakeoverTick, MovesToTheNextObstacleOnlyWhenTheHandBackIsOver) {
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  auto pilot = takeover();
  pilot->start(kGate07, course.size());
  EXPECT_TRUE(pilot->addReading("gate07", 0.5, 1.0));
  pilot->advance(nearGate(gates[kGate07], -1.0), course, kGate07);
  EXPECT_EQ(pilot->mode(), TakeoverMode::Active);
  // Past the gate, but not by the hand-back's 3 m.
  pilot->advance(nearGate(gates[kGate07], 2.9), course, kGate07 + 1);
  EXPECT_EQ(pilot->mode(), TakeoverMode::Active);
  EXPECT_EQ(pilot->obstacle(), 0u);
  pilot->advance(nearGate(gates[kGate07], 3.1), course, kGate07 + 1);
  EXPECT_EQ(pilot->mode(), TakeoverMode::HandingBack);
  for (int i = 0; i < 11; ++i) {
    pilot->advance(nearGate(gates[kGate07], 3.2), course, kGate07 + 1);
  }
  EXPECT_EQ(pilot->gain(), 0.0);
  EXPECT_EQ(pilot->mode(), TakeoverMode::Before);
  EXPECT_EQ(pilot->obstacle(), 1u);
  EXPECT_EQ(pilot->readings().newest(), -std::numeric_limits<double>::infinity());
}

TEST(TakeoverTick, WithHandbackNextHandsBackPastTheGateAfterTheBar) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover.yaml");
  doc["format_version"] = 5;
  doc["window"]["handback_next_m"] = 1.0;
  const auto later = TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover"));
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  Takeover pilot(later, std::vector<BarKind>{BarKind::UpDown, BarKind::Rotating}, BarSide::Right);
  pilot.start(kGate07, course.size());
  pilot.advance(nearGate(gates[kGate07], -1.0), course, kGate07);
  EXPECT_EQ(pilot.mode(), TakeoverMode::Active);
  // Past the bar's handback_m, which no longer hands back.
  pilot.advance(nearGate(gates[kGate07], 3.1), course, kGate07 + 1);
  EXPECT_EQ(pilot.mode(), TakeoverMode::Active);
  pilot.advance(nearGate(gates[kGate07 + 1], -0.5), course, kGate07 + 1);
  EXPECT_EQ(pilot.mode(), TakeoverMode::Active);
  // The gate after the bar passed, but not by 1 m.
  pilot.advance(nearGate(gates[kGate07 + 1], 0.9), course, kGate07 + 2);
  EXPECT_EQ(pilot.mode(), TakeoverMode::Active);
  pilot.advance(nearGate(gates[kGate07 + 1], 1.1), course, kGate07 + 2);
  EXPECT_EQ(pilot.mode(), TakeoverMode::HandingBack);
  for (int i = 0; i < 11; ++i) {
    pilot.advance(nearGate(gates[kGate07 + 1], 1.2), course, kGate07 + 2);
  }
  EXPECT_EQ(pilot.gain(), 0.0);
  EXPECT_EQ(pilot.mode(), TakeoverMode::Before);
  EXPECT_EQ(pilot.obstacle(), 1u);
}

TEST(TakeoverTick, WithOneBarStandingComesBackToItsGateEveryLap) {
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  Takeover pilot(
    networks(), std::vector<BarKind>{BarKind::LeftRight, BarKind::None}, BarSide::Left);
  pilot.start(kGate07 + 1, course.size());
  EXPECT_EQ(pilot.obstacle(), 0u);
  EXPECT_FALSE(pilot.addReading("gate12_oct", 0.5, 1.0));
  for (int lap = 0; lap < 3; ++lap) {
    for (int i = 0; i < 6; ++i) {
      pilot.advance(nearGate(gates[kGate07], -1.0), course, kGate07);
    }
    EXPECT_EQ(pilot.mode(), TakeoverMode::Active);
    EXPECT_GT(pilot.gain(), 0.0);
    EXPECT_TRUE(pilot.addReading("gate07", 0.5, 1.0 + lap));
    pilot.advance(nearGate(gates[kGate07], 3.1), course, kGate07 + 1);
    EXPECT_EQ(pilot.mode(), TakeoverMode::HandingBack);
    for (int i = 0; i < 11; ++i) {
      pilot.advance(nearGate(gates[kGate07], 3.2), course, kGate07 + 1);
    }
    EXPECT_EQ(pilot.mode(), TakeoverMode::Before);
    EXPECT_EQ(pilot.gain(), 0.0);
    EXPECT_EQ(pilot.obstacle(), 0u);
  }
}

TEST(TakeoverTick, NeverOpensAGateWithoutItsBar) {
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  Takeover pilot(
    networks(), std::vector<BarKind>{BarKind::UpDown, BarKind::None}, BarSide::Left);
  pilot.start(kGate12 - 1, course.size());
  EXPECT_EQ(pilot.obstacle(), 0u);
  for (const std::size_t target : {kGate12 - 1, kGate12, kGate12 + 1 - course.size()}) {
    pilot.advance(nearGate(gates[kGate12], -1.0), course, target);
    EXPECT_EQ(pilot.mode(), TakeoverMode::Before);
    EXPECT_EQ(pilot.gain(), 0.0);
  }
}

TEST(TakeoverTick, OpensGate12sWindowBeforeTheGateBeforeIt) {
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  auto pilot = takeover();
  pilot->start(kGate12, course.size());
  // At 5 m/s: 1.0 s out is outside its 0.9 s, 0.8 s inside.
  pilot->advance(nearGate(gates[kGate12], -5.0), course, kGate12 - 1);
  EXPECT_EQ(pilot->mode(), TakeoverMode::Before);
  pilot->advance(nearGate(gates[kGate12], -4.0), course, kGate12 - 1);
  EXPECT_EQ(pilot->mode(), TakeoverMode::Active);
}

TEST(TakeoverTick, ARecurrentAimRemembersItsApproachAloneAndForgetsAtTheNext) {
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  const auto main = mainPolicy();
  Takeover pilot(
    memoryNetworks(), std::vector<BarKind>{BarKind::UpDown, BarKind::Rotating}, BarSide::Right);
  pilot.start(0, course.size());
  ASSERT_EQ(pilot.obstacle(), 0u);
  // Gate07's approach starts with the gate before it the target.
  pilot.advance(nearGate(gates[kGate07 - 2], -3.0), course, kGate07 - 2);
  EXPECT_FALSE(pilot.near());
  pilot.act(*main, zeroRow(), nearGate(gates[kGate07 - 2], -3.0), course, 1.0);
  EXPECT_EQ(pilot.memory().norm(), 0.0);
  pilot.advance(nearGate(gates[kGate07 - 1], -3.0), course, kGate07 - 1);
  EXPECT_TRUE(pilot.near());
  EXPECT_EQ(pilot.gain(), 0.0);
  pilot.act(*main, zeroRow(), nearGate(gates[kGate07 - 1], -3.0), course, 1.01);
  EXPECT_GT(pilot.memory().norm(), 0.0);
  EXPECT_EQ(pilot.tick().row.size(), 0);
  // Through the window and the hand-back to gate12_oct's obstacle, which starts from nothing.
  pilot.advance(nearGate(gates[kGate07], -1.0), course, kGate07);
  pilot.advance(nearGate(gates[kGate07], -1.0), course, kGate07);
  ASSERT_GT(pilot.gain(), 0.0);
  pilot.act(*main, zeroRow(), nearGate(gates[kGate07], -1.0), course, 1.02);
  EXPECT_EQ(pilot.tick().row.size(), 72);
  pilot.advance(nearGate(gates[kGate07], 3.1), course, kGate07 + 1);
  for (int i = 0; i < 11; ++i) {
    pilot.advance(nearGate(gates[kGate07], 3.2), course, kGate07 + 1);
  }
  ASSERT_EQ(pilot.obstacle(), 1u);
  EXPECT_EQ(pilot.memory().norm(), 0.0);
  EXPECT_FALSE(pilot.near());
}

TEST(TakeoverTick, ShowsTheResidualWhatTheAimAppliesAndNothingPastThePlane) {
  const std::vector<rl_policy::Gate> gates = trainedGates();
  const Course course(gates);
  const auto main = mainPolicy();
  Takeover pilot(
    memoryNetworks(), std::vector<BarKind>{BarKind::UpDown, BarKind::Rotating}, BarSide::Right);
  pilot.start(kGate07, course.size());
  for (int i = 0; i < 8; ++i) {
    pilot.advance(nearGate(gates[kGate07], -1.0), course, kGate07);
  }
  ASSERT_EQ(pilot.gain(), 1.0);
  pilot.act(*main, zeroRow(), nearGate(gates[kGate07], -1.0), course, 1.0);
  const Eigen::VectorXd & row = pilot.tick().row;
  ASSERT_EQ(row.size(), 72);
  EXPECT_FLOAT_EQ(row(69), static_cast<float>(pilot.tick().aim.x()));
  EXPECT_FLOAT_EQ(row(70), static_cast<float>(pilot.tick().aim.y()));
  EXPECT_EQ(row(71), 1.0);
  EXPECT_NE(pilot.tick().aim, Eigen::Vector2d::Zero());
  pilot.advance(nearGate(gates[kGate07], 0.5), course, kGate07 + 1);
  pilot.act(*main, zeroRow(), nearGate(gates[kGate07], 0.5), course, 1.01);
  EXPECT_EQ(pilot.tick().row(69), 0.0);
  EXPECT_EQ(pilot.tick().row(70), 0.0);
}

namespace
{

// Flights of quadlab's env, tick by tick, against what the plugin's Takeover does with them; `held`
// counts the ticks a lane hold held.
void replayTrace(
  const std::string & fixture, const std::shared_ptr<const TakeoverNetworks> & files,
  std::size_t * held = nullptr)
{
  const YAML::Node trace = rl_policy_test::loadFixture(fixture);
  const double tolerance = trace["tolerance"].as<double>();
  const double delay = trace["delay_s"].as<double>();
  const double dt = trace["dt"].as<double>();
  const Course course(rl_policy_test::courseGates(trace["course"]));
  const auto main = mainPolicy();

  std::set<int> modes;
  std::set<int> obstacles;
  std::size_t window_ticks = 0;
  std::size_t readings = 0;
  std::size_t aimed_ticks = 0;
  std::size_t held_ticks = 0;
  for (const YAML::Node & flight : trace["flights"]) {
    std::vector<BarKind> kinds;
    for (const YAML::Node & kind : flight["kinds"]) {
      kinds.push_back(rl_policy::barKindFromString(kind.as<std::string>()));
    }
    Takeover pilot(files, kinds, BarSide::Right);
    const YAML::Node ticks = flight["ticks"];
    pilot.start(static_cast<std::size_t>(ticks[0][kTarget].as<double>()), course.size());
    std::size_t index = 0;
    for (const YAML::Node & node : ticks) {
      const std::vector<double> tick = rl_policy_test::doubles(node);
      const std::string where = "tick " + std::to_string(index++);
      const double t = tick[0];
      const auto target = static_cast<std::size_t>(tick[kTarget]);
      VehicleState state;
      state.position = Eigen::Map<const Eigen::Vector3d>(&tick[kPosition]);
      state.velocity = Eigen::Map<const Eigen::Vector3d>(&tick[kVelocity]);
      const Eigen::VectorXd row =
        Eigen::Map<const Eigen::VectorXd>(&tick[kRow], rl_policy::kMainRowSize);

      pilot.advance(state, course, target);
      ASSERT_EQ(static_cast<int>(pilot.mode()), static_cast<int>(tick[kMode])) << where;
      ASSERT_EQ(pilot.obstacle(), static_cast<std::size_t>(tick[kObstacle])) << where;
      ASSERT_NEAR(pilot.gain(), tick[kGain], tolerance) << where;
      if (tick[kReading] != 0.0) {
        if (files->spec().frame_hz > 0.0) {
          ASSERT_TRUE(pilot.addValues(t - delay, {tick[kValues], tick[kValues + 1]})) << where;
        } else {
          pilot.keepValues(t - delay, {tick[kValues], tick[kValues + 1]});
        }
        ++readings;
      }
      const Action action = pilot.act(*main, row, state, course, t);
      for (int k = 0; k < rl_policy::kActionSize; ++k) {
        ASSERT_NEAR(action[k], tick[kAction + k], tolerance) << where << " action " << k;
      }
      held_ticks += pilot.laneHold().on() ? 1 : 0;
      modes.insert(static_cast<int>(tick[kMode]));
      obstacles.insert(static_cast<int>(tick[kObstacle]));
      if (!(tick[kGain] > 0.0)) {
        ASSERT_EQ(tick.size(), kMain) << where;
        continue;
      }
      ++window_ticks;
      ASSERT_GT(tick.size(), kRowTail) << where;
      const rl_policy::TakeoverTick & done = pilot.tick();
      const Eigen::Index tail = rl_policy::kMainRowSize + rl_policy::kActionSize;
      ASSERT_EQ(static_cast<std::size_t>(done.row.size()), tick.size() - kRowTail + tail) << where;
      for (Eigen::Index i = tail; i < done.row.size(); ++i) {
        ASSERT_NEAR(done.row(i), tick[kRowTail + static_cast<std::size_t>(i - tail)], tolerance)
          << where << " row " << i;
      }
      for (int k = 0; k < rl_policy::kActionSize; ++k) {
        ASSERT_NEAR(done.main[k], tick[kMain + k], tolerance) << where << " main " << k;
        ASSERT_NEAR(done.aimed[k], tick[kAimed + k], tolerance) << where << " aimed " << k;
        ASSERT_NEAR(done.residual[k], tick[kResidual + k], tolerance) << where << " residual " << k;
      }
      // Past the plane the env still runs the aim and shifts by nothing.
      if (done.aim != Eigen::Vector2d::Zero()) {
        ++aimed_ticks;
        ASSERT_NEAR(done.aim.x(), tick[kAim], tolerance) << where;
        ASSERT_NEAR(done.aim.y(), tick[kAim + 1], tolerance) << where;
      }
    }
    EXPECT_GT(ticks.size(), static_cast<std::size_t>(5.0 / dt));
  }
  // The trace walked the whole machine, both obstacles and real readings. With two barred gates
  // the tail lasts no tick: the end of a hand-back is already the next obstacle's Before.
  EXPECT_EQ(modes, (std::set<int>{0, 1, 2}));
  EXPECT_EQ(obstacles, (std::set<int>{0, 1}));
  EXPECT_GT(window_ticks, 150u);
  EXPECT_GT(aimed_ticks, 100u);
  EXPECT_GT(readings, 15u);
  if (held != nullptr) {
    *held = held_ticks;
  }
}

}  // namespace

TEST(TakeoverTrace, IsWhatTheEnvFlew) {
  replayTrace("takeover_trace.yaml", networks());
}

TEST(TakeoverTrace, WithARecurrentAimIsWhatTheEnvFlewToo) {
  std::size_t held = 1;
  replayTrace("takeover_memory_trace.yaml", memoryNetworks(), &held);
  EXPECT_EQ(held, 0u);
}

// The trace's rotating bar is passed without a late flip: held, its aim flies as it says it.
TEST(TakeoverTrace, WithTheLaneHoldAndNoLateFlipIsWhatTheEnvFlewToo) {
  YAML::Node doc = rl_policy_test::loadFixture("takeover_memory.yaml");
  doc["format_version"] = 6;
  doc["window"]["lane_hold_s"] = 0.45;
  std::size_t held = 0;
  replayTrace(
    "takeover_memory_trace.yaml", TakeoverNetworks::load(rl_policy_test::writeYaml(doc, "takeover")),
    &held);
  EXPECT_GT(held, 20u);
}

// quadlab's TakeoverVecEnv._hold_lane on made-up approaches, tick by tick.
TEST(LaneHold, IsWhatTheEnvFlew) {
  const YAML::Node fixture = rl_policy_test::loadFixture("takeover_hold.yaml");
  const double hold_s = fixture["hold_s"].as<double>();
  EXPECT_EQ(fixture["min"].as<double>(), rl_policy::kHoldMin);
  std::set<std::string> reversing;
  std::size_t held_ticks = 0;
  for (const YAML::Node & lane : fixture["lanes"]) {
    const std::string name = lane["name"].as<std::string>();
    rl_policy::LaneHold hold;
    std::size_t index = 0;
    for (const YAML::Node & node : lane["ticks"]) {
      const std::vector<double> tick = rl_policy_test::doubles(node);
      const std::string where = name + " tick " + std::to_string(index++);
      const Eigen::Vector2d aim(tick[0], tick[1]);
      const bool on = tick[2] != 0.0 && tick[3] != 0.0 && tick[4] != 0.0;
      const Eigen::Vector2d held = hold.hold(aim, on, tick[5], hold_s);
      ASSERT_EQ(held.x(), tick[6]) << where;
      ASSERT_EQ(held.y(), tick[7]) << where;
      ASSERT_EQ(hold.on(), tick[8] != 0.0) << where;
      if (held != aim) {
        reversing.insert(name);
      }
      held_ticks += hold.on() ? 1 : 0;
    }
  }
  EXPECT_EQ(
    reversing,
    (std::set<std::string>{"late_flip", "small_aim", "turning_lane", "window_closing", "random"}));
  EXPECT_GT(held_ticks, 200u);
}

TEST(LaneHold, KeepsTheLaneItHeldOnAndEndsOffItsBar) {
  rl_policy::LaneHold hold;
  const Eigen::Vector2d over(0.0, 0.6);
  EXPECT_EQ(hold.hold(over, true, 0.5, 0.45), over);
  EXPECT_FALSE(hold.on());
  EXPECT_EQ(hold.hold(over, true, 0.45, 0.45), over);
  EXPECT_TRUE(hold.on());
  EXPECT_EQ(hold.hold(-over, true, 0.3, 0.45), over);
  // A small aim names no lane: it flies as held and leaves the reference.
  EXPECT_EQ(hold.hold(Eigen::Vector2d(0.0, -0.1), true, 0.2, 0.45), Eigen::Vector2d(0.0, 0.1));
  EXPECT_EQ(hold.hold(-over, true, 0.1, 0.45), over);
  EXPECT_EQ(hold.hold(-over, false, 0.05, 0.45), -over);
  EXPECT_FALSE(hold.on());
}

TEST_F(TakeoverMission, IsRefusedOnAHoverMission) {
  MissionController mission(config(MissionType::Hover), bank());
  EXPECT_THROW(mission.setTakeover(takeover()), std::invalid_argument);
}

TEST_F(TakeoverMission, IsRefusedForAnotherRacePolicy) {
  rl_policy::PolicySpec spec = mainPolicy()->spec();
  spec.source.checkpoint_sha256 = "another";
  PolicyBank policies;
  policies.add("race", std::make_shared<OtherPolicy>(spec));
  policies.add("hover", hover);
  MissionController mission(config(), std::move(policies));
  EXPECT_THROW(mission.setTakeover(takeover()), std::invalid_argument);
  EXPECT_EQ(mission.takeover(), nullptr);
}

TEST_F(TakeoverMission, IsRefusedForARacePolicyWithoutItsCircuit) {
  rl_policy::PolicySpec spec = mainPolicy()->spec();
  spec.course.reset();
  PolicyBank policies;
  policies.add("race", std::make_shared<OtherPolicy>(spec));
  policies.add("hover", hover);
  MissionController mission(config(), std::move(policies));
  EXPECT_THROW(mission.setTakeover(takeover()), std::invalid_argument);
}

TEST_F(TakeoverMission, IsRefusedForACircuitWhereItsGatesAreOthers) {
  rl_policy::PolicySpec spec = mainPolicy()->spec();
  spec.course->names[kGate07] = "gate07_moved";
  PolicyBank policies;
  policies.add("race", std::make_shared<OtherPolicy>(spec));
  policies.add("hover", hover);
  MissionController mission(config(), std::move(policies));
  EXPECT_THROW(mission.setTakeover(takeover()), std::invalid_argument);
}

TEST_F(TakeoverMission, OutsideTheWindowIsExactlyOneActOfMain) {
  MissionController mission(config(), bank());
  mission.setTakeover(takeover());
  ASSERT_TRUE(mission.start());
  const StepResult result = stepAt(mission, nearGate(mission.config().gates[0], -3.0));
  EXPECT_EQ(race->calls, 1u);
  ASSERT_TRUE(result.takeover.has_value());
  EXPECT_EQ(result.takeover->mode, TakeoverMode::Before);
  EXPECT_EQ(result.takeover->gain, 0.0);
  EXPECT_EQ(result.takeover->row.size(), 0);
  const Action alone = rl_policy::clip(mainPolicy()->act(result.observation));
  for (int k = 0; k < rl_policy::kActionSize; ++k) {
    EXPECT_EQ(result.action[k], alone[k]);
  }
}

TEST_F(TakeoverMission, InTheWindowMainRunsTwiceBeforeThePlaneAndOncePastIt) {
  MissionController mission(config(), bank());
  mission.setTakeover(takeover());
  ASSERT_TRUE(mission.start());
  passGates(mission, kGate07);
  EXPECT_EQ(mission.takeover()->mode(), TakeoverMode::Active);
  const rl_policy::Gate gate = mission.config().gates[kGate07];

  stepAt(mission, nearGate(gate, -0.6));
  race->calls = 0;
  const StepResult before = stepAt(mission, nearGate(gate, -0.5));
  EXPECT_EQ(race->calls, 2u);
  ASSERT_TRUE(before.takeover.has_value());
  EXPECT_GT(before.takeover->gain, 0.0);
  EXPECT_NE(before.takeover->aim, Eigen::Vector2d::Zero());
  // The history holds what flew, not main's alone.
  const Action flown = before.action;

  race->calls = 0;
  const StepResult after = stepAt(mission, nearGate(gate, 0.1));
  EXPECT_EQ(race->calls, 1u);
  ASSERT_TRUE(after.takeover.has_value());
  EXPECT_EQ(after.takeover->aim, Eigen::Vector2d::Zero());
  EXPECT_EQ(after.takeover->aimed, after.takeover->main);
  const int history = rl_policy::gateRelativeSize(rl_policy::Attitude::R6);
  for (int k = 0; k < rl_policy::kActionSize; ++k) {
    EXPECT_EQ(after.observation(history + k), static_cast<float>(flown[k]));
  }
}

TEST_F(TakeoverMission, OnlyARaceTakesReadings) {
  MissionController mission(config(), bank());
  mission.setTakeover(takeover());
  EXPECT_FALSE(mission.addBarReading("gate07", 0.5, 1.0));
  ASSERT_TRUE(mission.start());
  EXPECT_TRUE(mission.addBarReading("gate07", 0.5, 1.0));
  EXPECT_TRUE(mission.requestHold(5.0, 4.0, 1.5, 0.0));
  EXPECT_FALSE(mission.addBarReading("gate07", 0.5, 2.0));
}

TEST_F(TakeoverMission, TheHoldNeverRunsTheNetworks) {
  MissionController mission(config(), bank());
  mission.setTakeover(takeover());
  ASSERT_TRUE(mission.start());
  EXPECT_TRUE(mission.requestHold(5.0, 4.0, 1.5, 0.0));
  EXPECT_EQ(mission.phase(), Phase::Hold);
  race->calls = 0;
  hover->calls = 0;
  const StepResult result =
    stepAt(mission, at(Eigen::Vector3d(5.0, 4.0, 1.5), Eigen::Vector3d::Zero()));
  EXPECT_EQ(hover->calls, 1u);
  EXPECT_EQ(race->calls, 0u);
  EXPECT_FALSE(result.takeover.has_value());
}

TEST_F(TakeoverMission, WithoutOneARaceStepIsWhatItAlwaysWas) {
  MissionController mission(config(), bank());
  ASSERT_TRUE(mission.start());
  EXPECT_EQ(mission.takeover(), nullptr);
  EXPECT_FALSE(mission.addBarReading("gate07", 0.5, 1.0));
  const StepResult result = stepAt(mission, nearGate(mission.config().gates[0], -3.0));
  EXPECT_EQ(race->calls, 1u);
  EXPECT_FALSE(result.takeover.has_value());
}
