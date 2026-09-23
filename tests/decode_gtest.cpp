#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "as2_rl_policy/core/decode.hpp"
#include "fixture_utils.hpp"

namespace
{

rl_policy::ActionSpec specFromFixture(const YAML::Node & fixture)
{
  rl_policy::ActionSpec spec;
  spec.contract = rl_policy::kBodyRatesContract;
  spec.specific_thrust_max_ms2 = fixture["specific_thrust_max_ms2"].as<double>();
  const auto rates = rl_policy_test::doubles(fixture["max_body_rate_rad_s"]);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    spec.max_body_rate_rad_s[axis] = rates.at(axis);
  }
  return spec;
}

}  // namespace

TEST(Decode, MatchesTheFixture) {
  const YAML::Node fixture = rl_policy_test::loadFixture("decode.yaml");
  const rl_policy::ActionSpec spec = specFromFixture(fixture);
  const double mass = fixture["mass_kg"].as<double>();
  const double tolerance = fixture["tolerance"].as<double>();
  const YAML::Node rows = fixture["rows"];
  ASSERT_GT(rows.size(), 0u);
  bool out_of_range = false;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const auto values = rl_policy_test::doubles(rows[r]["action"]);
    const rl_policy::Action action{values[0], values[1], values[2], values[3]};
    for (const double value : action) {
      out_of_range = out_of_range || std::abs(value) > 1.0;
    }
    const rl_policy::Command command = rl_policy::decode(action, spec, mass);
    EXPECT_NEAR(command.thrust_n, rows[r]["thrust_n"].as<double>(), tolerance) << "row " << r;
    const auto rates = rl_policy_test::doubles(rows[r]["rates"]);
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_NEAR(command.rates(axis), rates[axis], tolerance) << "row " << r << " axis " << axis;
    }
  }
  EXPECT_TRUE(out_of_range) << "the fixture has no out-of-range action";
}

TEST(Decode, ClipsFirst) {
  rl_policy::ActionSpec spec;
  spec.specific_thrust_max_ms2 = 40.0;
  spec.max_body_rate_rad_s = {10.0, 20.0, 30.0};
  const rl_policy::Command over = rl_policy::decode({1.7, -3.0, 2.0, 0.5}, spec, 2.0);
  EXPECT_DOUBLE_EQ(over.thrust_n, 80.0);
  EXPECT_DOUBLE_EQ(over.rates(0), -10.0);
  EXPECT_DOUBLE_EQ(over.rates(1), 20.0);
  EXPECT_DOUBLE_EQ(over.rates(2), 15.0);
  const rl_policy::Command idle = rl_policy::decode({-1.2, 0.0, 0.0, 0.0}, spec, 2.0);
  EXPECT_DOUBLE_EQ(idle.thrust_n, 0.0);
  EXPECT_EQ(rl_policy::clip({-2.0, 2.0, 0.25, -0.25}), (rl_policy::Action{-1.0, 1.0, 0.25, -0.25}));
}
