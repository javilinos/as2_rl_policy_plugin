#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "as2_rl_policy/core/angles.hpp"
#include "as2_rl_policy/core/course.hpp"
#include "as2_rl_policy/core/observation.hpp"
#include "fixture_utils.hpp"

namespace
{

using rl_policy::ActionHistory;
using rl_policy::Course;
using rl_policy_test::doubles;

ActionHistory historyFromRow(const std::vector<double> & values, std::size_t length)
{
  ActionHistory history(length);
  for (std::size_t age = length; age-- > 0; ) {
    rl_policy::Action action{};
    for (std::size_t k = 0; k < 4; ++k) {
      action[k] = values[4 * age + k];
    }
    history.push(action);
  }
  return history;
}

// Every fixture row rebuilt from its state and compared with the trainer's observation, in the
// fixture's layout (the Euler one when it names none).
void checkFixture(const std::string & name)
{
  const YAML::Node fixture = rl_policy_test::loadFixture(name);
  const Course course(rl_policy_test::courseGates(fixture["course"]));
  const double normalization = fixture["motor_speed_normalization"].as<double>();
  const std::size_t length = fixture["action_history"].as<std::size_t>();
  const double tolerance = fixture["tolerance"].as<double>();
  const rl_policy::Attitude attitude = rl_policy::layoutAttitude(
    fixture["layout"] ? fixture["layout"].as<std::string>() : rl_policy::kGateRelativeLayout);
  const YAML::Node rows = fixture["rows"];
  ASSERT_GT(rows.size(), 0u);

  std::set<std::size_t> targets;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const auto state = rl_policy_test::stateFromRow(doubles(rows[r]["state"]));
    const std::size_t target = rows[r]["target"].as<std::size_t>();
    const ActionHistory history = historyFromRow(doubles(rows[r]["history"]), length);
    const auto expected = doubles(rows[r]["obs"]);
    targets.insert(target % course.size());

    const Eigen::VectorXd obs = rl_policy::buildObservation(
      state, course, target, history, normalization, attitude);
    ASSERT_EQ(static_cast<std::size_t>(obs.size()), expected.size()) << name << " row " << r;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(obs(static_cast<Eigen::Index>(i)), expected[i], tolerance)
        << name << " row " << r << " entry " << i << " target " << target;
    }
  }
  EXPECT_EQ(targets.size(), course.size()) << name << " does not cover every target";
}

}  // namespace

TEST(Observation, SeasonTwoRowsMatchTheKernel) {
  checkFixture("obs_season2.yaml");
}

TEST(Observation, HoverRowsMatchTheKernel) {
  checkFixture("obs_hover.yaml");
}

TEST(Observation, SeasonTwoR6RowsMatchTheTrainer) {
  checkFixture("obs_season2_r6.yaml");
}

// The R6 entries are the gate-frame body x and thrust axes of the quaternion itself, whatever its
// Euler angles, and the rest of the row is the Euler row's, shifted by three.
TEST(Observation, R6IsTheGateFrameRotationOfTheQuaternion) {
  const YAML::Node fixture = rl_policy_test::loadFixture("obs_season2_r6.yaml");
  const Course course(rl_policy_test::courseGates(fixture["course"]));
  const YAML::Node rows = fixture["rows"];
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const auto state = rl_policy_test::stateFromRow(doubles(rows[r]["state"]));
    const std::size_t target = rows[r]["target"].as<std::size_t>();
    const Eigen::VectorXd r6 = rl_policy::buildObservation(
      state, course, target, ActionHistory(2), 3000.0, rl_policy::Attitude::R6);
    const Eigen::VectorXd euler = rl_policy::buildObservation(
      state, course, target, ActionHistory(2), 3000.0, rl_policy::Attitude::Euler);
    ASSERT_EQ(r6.size(), 31);
    ASSERT_EQ(euler.size(), 28);

    const Eigen::Matrix3d gate_frame =
      (Eigen::AngleAxisd(-course.gate(target).yaw, Eigen::Vector3d::UnitZ()) *
      state.orientation.normalized()).toRotationMatrix();
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(r6(6 + k), gate_frame(k, 0), 1e-6) << "row " << r << " body x " << k;
      EXPECT_NEAR(r6(9 + k), gate_frame(k, 2), 1e-6) << "row " << r << " thrust axis " << k;
    }
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(r6(i), euler(i)) << "row " << r << " entry " << i;
    }
    for (int i = 9; i < 28; ++i) {
      EXPECT_EQ(r6(i + 3), euler(i)) << "row " << r << " entry " << i;
    }
  }
}

// Across the roll's wrap the Euler row jumps by 2 pi and the R6 row does not.
TEST(Observation, R6IsContinuousWhereTheRollWraps) {
  const Course course(std::vector<rl_policy::Gate>{rl_policy::Gate()});
  rl_policy::VehicleState before;
  rl_policy::VehicleState after;
  const double pi = rl_policy::kPi;
  before.orientation = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(pi - 1e-4, Eigen::Vector3d::UnitX());
  after.orientation = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(pi + 1e-4, Eigen::Vector3d::UnitX());
  const auto observe = [&](const rl_policy::VehicleState & state, rl_policy::Attitude attitude) {
      return rl_policy::buildObservation(state, course, 0, ActionHistory(2), 3000.0, attitude);
    };
  const Eigen::VectorXd euler_before = observe(before, rl_policy::Attitude::Euler);
  const Eigen::VectorXd euler_after = observe(after, rl_policy::Attitude::Euler);
  EXPECT_GT(std::abs(euler_before(6) - euler_after(6)), 6.0);
  const Eigen::VectorXd r6_before = observe(before, rl_policy::Attitude::R6);
  const Eigen::VectorXd r6_after = observe(after, rl_policy::Attitude::R6);
  EXPECT_LT((r6_before.segment<6>(6) - r6_after.segment<6>(6)).cwiseAbs().maxCoeff(), 1e-3);
  EXPECT_LT(r6_before(11), -0.9) << "the thrust axis points down, inverted";
}

TEST(Observation, R6HistoryFollowsTheTwentyThreeEntries) {
  ActionHistory history(2);
  const rl_policy::Action older{0.1, 0.2, 0.3, 0.4};
  const rl_policy::Action newer{-0.5, -0.6, -0.7, -0.8};
  history.push(older);
  history.push(newer);
  const Course course(std::vector<rl_policy::Gate>{rl_policy::Gate()});
  const Eigen::VectorXd obs = rl_policy::buildObservation(
    rl_policy::VehicleState(), course, 0, history, 3000.0, rl_policy::Attitude::R6);
  ASSERT_EQ(obs.size(), 31);
  for (int k = 0; k < 4; ++k) {
    EXPECT_EQ(obs(23 + k), static_cast<double>(static_cast<float>(newer[k])));
    EXPECT_EQ(obs(27 + k), static_cast<double>(static_cast<float>(older[k])));
  }
  // Level and facing the gate: the body x axis along the gate's x, the thrust axis up.
  const std::vector<double> level = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  for (int k = 0; k < 6; ++k) {
    EXPECT_NEAR(obs(6 + k), level[k], 1e-7) << "entry " << 6 + k;
  }
}

TEST(Observation, LayoutsAreTheKernelsAndR6) {
  EXPECT_EQ(rl_policy::layoutAttitude("gate_relative_20"), rl_policy::Attitude::Euler);
  EXPECT_EQ(rl_policy::layoutAttitude("gate_relative_r6_23"), rl_policy::Attitude::R6);
  EXPECT_EQ(rl_policy::gateRelativeSize(rl_policy::Attitude::Euler), 20);
  EXPECT_EQ(rl_policy::gateRelativeSize(rl_policy::Attitude::R6), 23);
  EXPECT_THROW(rl_policy::layoutAttitude("gate_relative_30"), rl_policy::PolicyError);
}

TEST(Observation, OneGateCourseHasNoNextGate) {
  const YAML::Node fixture = rl_policy_test::loadFixture("obs_hover.yaml");
  const Course course(rl_policy_test::courseGates(fixture["course"]));
  ASSERT_EQ(course.size(), 1u);
  const YAML::Node rows = fixture["rows"];
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const auto state = rl_policy_test::stateFromRow(doubles(rows[r]["state"]));
    const Eigen::VectorXd obs = rl_policy::buildObservation(
      state, course, 0, ActionHistory(2), fixture["motor_speed_normalization"].as<double>());
    for (int i = 16; i < 20; ++i) {
      EXPECT_EQ(obs(i), 0.0) << "row " << r << " entry " << i;
    }
  }
  rl_policy::Gate gate;
  gate.position = Eigen::Vector3d(3.0, -2.0, 1.5);
  gate.yaw = 2.5;
  EXPECT_EQ(Course::single(gate).nextBlock(0), Eigen::Vector4d::Zero());
}

TEST(Observation, QuaternionIsNormalized) {
  const YAML::Node fixture = rl_policy_test::loadFixture("obs_season2.yaml");
  const Course course(rl_policy_test::courseGates(fixture["course"]));
  auto state = rl_policy_test::stateFromRow(doubles(fixture["rows"][0]["state"]));
  const Eigen::VectorXd unit =
    rl_policy::buildObservation(state, course, 0, ActionHistory(2), 3000.0);
  state.orientation.coeffs() *= 1.7;
  const Eigen::VectorXd scaled =
    rl_policy::buildObservation(state, course, 0, ActionHistory(2), 3000.0);
  EXPECT_LT((unit - scaled).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(Observation, HistoryIsNewestFirst) {
  ActionHistory history(2);
  const rl_policy::Action older{0.1, 0.2, 0.3, 0.4};
  const rl_policy::Action newer{-0.5, -0.6, -0.7, -0.8};
  history.push(older);
  history.push(newer);
  EXPECT_EQ(history[0], newer);
  EXPECT_EQ(history[1], older);

  const Course course(std::vector<rl_policy::Gate>{rl_policy::Gate()});
  const Eigen::VectorXd obs = rl_policy::buildObservation(
    rl_policy::VehicleState(), course, 0, history, 3000.0);
  ASSERT_EQ(obs.size(), 28);
  for (int k = 0; k < 4; ++k) {
    EXPECT_EQ(obs(20 + k), static_cast<double>(static_cast<float>(newer[k])));
    EXPECT_EQ(obs(24 + k), static_cast<double>(static_cast<float>(older[k])));
  }

  history.push(rl_policy::Action{1.0, 1.0, 1.0, 1.0});
  EXPECT_EQ(history[1], newer);
}

TEST(Observation, ZeroClearsTheHistory) {
  ActionHistory history(2);
  history.push(rl_policy::Action{0.1, 0.2, 0.3, 0.4});
  history.zero();
  for (std::size_t age = 0; age < history.size(); ++age) {
    EXPECT_EQ(history[age], (rl_policy::Action{0.0, 0.0, 0.0, 0.0}));
  }
  ActionHistory none;
  none.push(rl_policy::Action{0.1, 0.2, 0.3, 0.4});
  EXPECT_EQ(none.size(), 0u);
}

TEST(Observation, EntriesAreFloat32) {
  const YAML::Node fixture = rl_policy_test::loadFixture("obs_season2.yaml");
  const Course course(rl_policy_test::courseGates(fixture["course"]));
  const auto state = rl_policy_test::stateFromRow(doubles(fixture["rows"][1]["state"]));
  const Eigen::VectorXd obs =
    rl_policy::buildObservation(state, course, 1, ActionHistory(2), 3000.0);
  for (Eigen::Index i = 0; i < obs.size(); ++i) {
    EXPECT_EQ(obs(i), static_cast<double>(static_cast<float>(obs(i))));
  }
}

TEST(Observation, WrapPiIsTheKernels) {
  const double pi = rl_policy::kPi;
  EXPECT_DOUBLE_EQ(rl_policy::wrapPi(0.0), 0.0);
  EXPECT_DOUBLE_EQ(rl_policy::wrapPi(pi), -pi);
  EXPECT_DOUBLE_EQ(rl_policy::wrapPi(-pi), -pi);
  EXPECT_NEAR(rl_policy::wrapPi(3.0 * pi + 0.25), -pi + 0.25, 1e-12);
  EXPECT_NEAR(rl_policy::wrapPi(-2.0 * pi - 0.25), -0.25, 1e-12);
  for (double angle = -20.0; angle < 20.0; angle += 0.37) {
    const double wrapped = rl_policy::wrapPi(angle);
    EXPECT_GE(wrapped, -pi);
    EXPECT_LT(wrapped, pi);
    EXPECT_NEAR(std::remainder(wrapped - angle, 2.0 * pi), 0.0, 1e-9);
  }
}

TEST(Observation, RelativeTablesLoop) {
  const YAML::Node fixture = rl_policy_test::loadFixture("obs_season2.yaml");
  const auto gates = rl_policy_test::courseGates(fixture["course"]);
  const Course course(gates);
  const std::size_t n = course.size();
  ASSERT_GT(n, 1u);
  EXPECT_EQ(course.next(n - 1), 0u);
  const Eigen::Vector3d delta = gates[0].position - gates[n - 1].position;
  const double c = std::cos(gates[n - 1].yaw);
  const double s = std::sin(gates[n - 1].yaw);
  const Eigen::Vector4d wrap = course.nextBlock(n - 1);
  EXPECT_NEAR(wrap(0), c * delta.x() + s * delta.y(), 1e-12);
  EXPECT_NEAR(wrap(1), -s * delta.x() + c * delta.y(), 1e-12);
  EXPECT_NEAR(wrap(2), delta.z(), 1e-12);
  EXPECT_NEAR(wrap(3), rl_policy::wrapPi(gates[0].yaw - gates[n - 1].yaw), 1e-12);
}
