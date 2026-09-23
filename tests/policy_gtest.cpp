#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "as2_rl_policy/core/policy.hpp"
#include "fixture_utils.hpp"

namespace
{

using rl_policy::MlpPolicy;
using rl_policy::PolicyBank;
using rl_policy::PolicyError;
using rl_policy_test::fixturePath;
using rl_policy_test::writeYaml;

const char kPolicyFixture[] = "policy_random.yaml";

YAML::Node policyYaml()
{
  return YAML::Clone(rl_policy_test::loadFixture(kPolicyFixture));
}

// The loader's message, or "" when the file loads.
std::string refusal(const YAML::Node & node)
{
  try {
    MlpPolicy::load(writeYaml(node, "policy"));
  } catch (const PolicyError & e) {
    return e.what();
  }
  return "";
}

class StubPolicy : public rl_policy::Policy
{
public:
  explicit StubPolicy(rl_policy::PolicySpec spec)
  : spec_(std::move(spec)) {}

  rl_policy::Action act(const Eigen::VectorXd &) const override {return {};}

  const rl_policy::PolicySpec & spec() const override {return spec_;}

private:
  rl_policy::PolicySpec spec_;
};

rl_policy::PolicySpec supportedSpec(double dt)
{
  rl_policy::PolicySpec spec;
  spec.task = "race";
  spec.dt = dt;
  spec.action.contract = rl_policy::kBodyRatesContract;
  spec.action.specific_thrust_max_ms2 = 40.0;
  spec.action.max_body_rate_rad_s = {20.0, 20.0, 20.0};
  spec.observation.layout = rl_policy::kGateRelativeLayout;
  spec.observation.action_history = 2;
  spec.observation.dim = 28;
  spec.observation.motor_speed_normalization = 3000.0;
  return spec;
}

}  // namespace

TEST(PolicyFile, LoadsAndReproducesItsGoldens) {
  const auto policy = MlpPolicy::load(fixturePath(kPolicyFixture));
  const rl_policy::PolicySpec & spec = policy->spec();
  EXPECT_EQ(spec.observation.layout, rl_policy::kGateRelativeLayout);
  EXPECT_EQ(spec.observation.dim, 20 + 4 * spec.observation.action_history);
  EXPECT_EQ(spec.action.contract, rl_policy::kBodyRatesContract);
  EXPECT_GT(spec.dt, 0.0);
  EXPECT_GE(policy->goldenCheck().rows, 1u);
  EXPECT_LE(policy->goldenCheck().max_error, policy->goldenCheck().tolerance);

  // Checked here too, independently of the loader.
  const YAML::Node golden = rl_policy_test::loadFixture(kPolicyFixture)["golden"];
  const double tolerance = golden["tolerance"].as<double>();
  ASSERT_EQ(golden["obs"].size(), golden["action"].size());
  for (std::size_t row = 0; row < golden["obs"].size(); ++row) {
    const auto obs = rl_policy_test::doubles(golden["obs"][row]);
    const auto expected = rl_policy_test::doubles(golden["action"][row]);
    const rl_policy::Action action = policy->act(
      Eigen::Map<const Eigen::VectorXd>(obs.data(), static_cast<Eigen::Index>(obs.size())));
    for (int k = 0; k < 4; ++k) {
      EXPECT_NEAR(action[k], expected[k], tolerance) << "row " << row << " entry " << k;
    }
  }
}

TEST(PolicyFile, ActionIsClippedToTheUnitBox) {
  const auto policy = MlpPolicy::load(fixturePath(kPolicyFixture));
  for (const double scale : {-1.0e3, 1.0e3}) {
    const int dim = policy->spec().observation.dim;
    const Eigen::VectorXd obs = Eigen::VectorXd::LinSpaced(dim, 1.0, -2.0) * scale;
    for (const double value : policy->act(obs)) {
      EXPECT_GE(value, -1.0);
      EXPECT_LE(value, 1.0);
    }
  }
  EXPECT_THROW(policy->act(Eigen::VectorXd::Zero(3)), std::invalid_argument);
}

TEST(PolicyFile, RefusesACorruptedWeight) {
  YAML::Node node = policyYaml();
  YAML::Node weight = node["network"]["layers"][3]["weight"];
  weight[3] = weight[3].as<double>() + 1.0;
  const std::string message = refusal(node);
  EXPECT_NE(message.find("golden row"), std::string::npos) << message;
}

TEST(PolicyFile, RefusesAWrongShape) {
  {
    YAML::Node node = policyYaml();
    node["network"]["layers"][0]["shape"][1] = 27;
    const std::string message = refusal(node);
    EXPECT_NE(message.find("values, shape"), std::string::npos) << message;
  }
  {
    YAML::Node node = policyYaml();
    node["network"]["layers"][1]["bias"].push_back(0.5);
    const std::string message = refusal(node);
    EXPECT_NE(message.find("bias"), std::string::npos) << message;
  }
  {
    YAML::Node node = policyYaml();
    node["observation"]["dim"] = 27;
    const std::string message = refusal(node);
    EXPECT_NE(message.find("observation.dim"), std::string::npos) << message;
  }
  {
    YAML::Node node = policyYaml();
    node["observation"]["action_history"] = 1;
    node["observation"]["dim"] = 24;
    const std::string message = refusal(node);
    EXPECT_NE(message.find("layer 0 takes 28 inputs"), std::string::npos) << message;
  }
  {
    YAML::Node node = policyYaml();
    YAML::Node last = node["network"]["layers"][3];
    YAML::Node weight(YAML::NodeType::Sequence);
    for (std::size_t i = 0; i < 3 * 64; ++i) {
      weight.push_back(last["weight"][i]);
    }
    YAML::Node bias(YAML::NodeType::Sequence);
    for (std::size_t i = 0; i < 3; ++i) {
      bias.push_back(last["bias"][i]);
    }
    last["shape"][0] = 3;
    last["weight"] = weight;
    last["bias"] = bias;
    const std::string message = refusal(node);
    EXPECT_NE(message.find("the last layer gives 3 outputs"), std::string::npos) << message;
  }
}

TEST(PolicyFile, RefusesAWrongFormat) {
  const std::vector<std::pair<std::vector<std::string>, std::string>> edits = {
    {{"format"}, "onnx"},
    {{"observation", "layout"}, "gate_relative_30"},
    {{"network", "activation"}, "relu"},
    {{"network", "output"}, "tanh"},
    {{"action", "contract"}, "motors_v1"},
  };
  for (const auto & [keys, value] : edits) {
    YAML::Node node = policyYaml();
    if (keys.size() == 1) {
      node[keys[0]] = value;
    } else {
      node[keys[0]][keys[1]] = value;
    }
    const std::string message = refusal(node);
    EXPECT_NE(message.find(value), std::string::npos) << keys.back() << ": " << message;
  }
  {
    YAML::Node node = policyYaml();
    node["format_version"] = 2;
    EXPECT_NE(refusal(node).find("format_version 2"), std::string::npos);
  }
  {
    YAML::Node node = policyYaml();
    node.remove("golden");
    EXPECT_NE(refusal(node).find("missing 'golden'"), std::string::npos);
  }
  {
    YAML::Node node = policyYaml();
    node["golden"]["obs"] = YAML::Node(YAML::NodeType::Sequence);
    node["golden"]["action"] = YAML::Node(YAML::NodeType::Sequence);
    EXPECT_NE(refusal(node).find("no rows"), std::string::npos);
  }
  EXPECT_THROW(MlpPolicy::load(fixturePath("does_not_exist.yaml")), PolicyError);
  const std::string not_yaml = rl_policy_test::temporaryFile("not_yaml");
  std::ofstream(not_yaml) << "format: [unclosed\n";
  EXPECT_THROW(MlpPolicy::load(not_yaml), PolicyError);
}

TEST(PolicyFile, TaskDefaultsToRace) {
  YAML::Node node = policyYaml();
  node.remove("task");
  const auto policy = MlpPolicy::load(writeYaml(node, "policy_no_task"));
  EXPECT_EQ(policy->spec().task, "race");
}

TEST(PolicyBank, SharesOneDt) {
  const std::string file = fixturePath(kPolicyFixture);
  const PolicyBank bank = PolicyBank::load({"race", "hover"}, {file, file});
  EXPECT_EQ(bank.size(), 2u);
  EXPECT_TRUE(bank.has("race"));
  EXPECT_TRUE(bank.has("hover"));
  EXPECT_FALSE(bank.has("other"));
  EXPECT_DOUBLE_EQ(bank.dt(), bank.get("race")->spec().dt);
  EXPECT_THROW(bank.get("other"), PolicyError);

  YAML::Node node = policyYaml();
  node["dt"] = 0.02;
  const std::string slower = writeYaml(node, "policy_slow");
  EXPECT_THROW(PolicyBank::load({"race", "hover"}, {file, slower}), PolicyError);

  PolicyBank stubs;
  stubs.add("race", std::make_shared<StubPolicy>(supportedSpec(0.01)));
  EXPECT_THROW(stubs.add("hover", std::make_shared<StubPolicy>(supportedSpec(0.02))), PolicyError);
  EXPECT_NO_THROW(stubs.add("hover", std::make_shared<StubPolicy>(supportedSpec(0.01))));
}

TEST(PolicyBank, RefusesAnUnsupportedLayout) {
  PolicyBank bank;
  rl_policy::PolicySpec spec = supportedSpec(0.01);
  spec.observation.layout = "gate_relative_30";
  EXPECT_THROW(bank.add("race", std::make_shared<StubPolicy>(spec)), PolicyError);
  spec = supportedSpec(0.01);
  spec.observation.dim = 30;
  EXPECT_THROW(bank.add("race", std::make_shared<StubPolicy>(spec)), PolicyError);
  spec = supportedSpec(0.01);
  spec.action.contract = "motors_v1";
  EXPECT_THROW(bank.add("race", std::make_shared<StubPolicy>(spec)), PolicyError);
  EXPECT_EQ(bank.size(), 0u);
}

TEST(PolicyBank, RefusesABadList) {
  const std::string file = fixturePath(kPolicyFixture);
  EXPECT_THROW(PolicyBank::load({"race"}, {file, file}), PolicyError);
  EXPECT_THROW(PolicyBank::load({}, {}), PolicyError);
  EXPECT_THROW(PolicyBank::load({"race", "race"}, {file, file}), PolicyError);
  EXPECT_THROW(PolicyBank::load({"race"}, {""}), PolicyError);
  EXPECT_THROW(PolicyBank::load({""}, {file}), PolicyError);
}
