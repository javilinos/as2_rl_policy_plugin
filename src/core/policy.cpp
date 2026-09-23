#include "as2_rl_policy/core/policy.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace rl_policy
{

namespace
{

// Two policies share a step when their dt agree to a nanosecond.
constexpr double kDtTolerance = 1.0e-9;

std::string number(double value)
{
  std::ostringstream stream;
  stream.precision(9);
  stream << value;
  return stream.str();
}

std::string join(const std::string & path, const std::string & key)
{
  return path.empty() ? key : path + "." + key;
}

YAML::Node required(const YAML::Node & parent, const std::string & key, const std::string & path)
{
  if (!parent.IsMap()) {
    throw PolicyError("'" + (path.empty() ? std::string("the file") : path) + "' is not a mapping");
  }
  const YAML::Node value = parent[key];
  if (!value.IsDefined() || value.IsNull()) {
    throw PolicyError("missing '" + join(path, key) + "'");
  }
  return value;
}

template<typename T>
T scalar(const YAML::Node & node, const std::string & path, const char * type)
{
  if (!node.IsScalar()) {
    throw PolicyError("'" + path + "' is not a " + type);
  }
  try {
    return node.as<T>();
  } catch (const YAML::Exception &) {
    throw PolicyError("'" + path + "' is not a " + type + ": '" + node.Scalar() + "'");
  }
}

double finiteDouble(const YAML::Node & node, const std::string & path)
{
  const double value = scalar<double>(node, path, "number");
  if (!std::isfinite(value)) {
    throw PolicyError("'" + path + "' is not finite");
  }
  return value;
}

double positiveDouble(const YAML::Node & node, const std::string & path)
{
  const double value = finiteDouble(node, path);
  if (!(value > 0.0)) {
    throw PolicyError("'" + path + "' must be positive, got " + number(value));
  }
  return value;
}

std::vector<double> doubles(const YAML::Node & node, const std::string & path)
{
  if (!node.IsSequence()) {
    throw PolicyError("'" + path + "' is not a sequence");
  }
  std::vector<double> values;
  values.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    values.push_back(finiteDouble(node[i], path + "[" + std::to_string(i) + "]"));
  }
  return values;
}

std::string optionalString(
  const YAML::Node & parent, const std::string & key, const std::string & fallback)
{
  if (!parent.IsMap()) {
    return fallback;
  }
  const YAML::Node value = parent[key];
  if (!value.IsDefined() || value.IsNull() || !value.IsScalar()) {
    return fallback;
  }
  return value.Scalar();
}

void expectString(
  const YAML::Node & parent, const std::string & key, const std::string & path,
  const std::string & expected)
{
  const std::string value =
    scalar<std::string>(required(parent, key, path), join(path, key), "string");
  if (value != expected) {
    throw PolicyError(
            "'" + join(path, key) + "' is '" + value + "', only '" + expected + "' is supported");
  }
}

// Weights are float32 values; parsing them as float keeps exactly the network that was trained.
double float32(double value)
{
  return static_cast<double>(static_cast<float>(value));
}

MlpPolicy::Layer parseLayer(const YAML::Node & node, const std::string & path)
{
  const YAML::Node shape_node = required(node, "shape", path);
  if (!shape_node.IsSequence() || shape_node.size() != 2) {
    throw PolicyError("'" + path + ".shape' must be [out, in]");
  }
  const int rows = scalar<int>(shape_node[0], path + ".shape[0]", "integer");
  const int cols = scalar<int>(shape_node[1], path + ".shape[1]", "integer");
  if (rows <= 0 || cols <= 0) {
    throw PolicyError("'" + path + ".shape' must be positive");
  }
  const std::vector<double> weight = doubles(required(node, "weight", path), path + ".weight");
  const std::vector<double> bias = doubles(required(node, "bias", path), path + ".bias");
  if (weight.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {
    throw PolicyError(
            "'" + path + ".weight' has " + std::to_string(weight.size()) + " values, shape [" +
            std::to_string(rows) + ", " + std::to_string(cols) + "] needs " +
            std::to_string(rows * cols));
  }
  if (bias.size() != static_cast<std::size_t>(rows)) {
    throw PolicyError(
            "'" + path + ".bias' has " + std::to_string(bias.size()) + " values, shape needs " +
            std::to_string(rows));
  }
  MlpPolicy::Layer layer;
  layer.weight.resize(rows, cols);
  layer.bias.resize(rows);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      layer.weight(r, c) = float32(weight[static_cast<std::size_t>(r) * cols + c]);
    }
    layer.bias(r) = float32(bias[r]);
  }
  return layer;
}

std::vector<std::vector<double>> rows(
  const YAML::Node & node, const std::string & path, std::size_t width)
{
  if (!node.IsSequence()) {
    throw PolicyError("'" + path + "' is not a sequence");
  }
  std::vector<std::vector<double>> out;
  out.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string row_path = path + "[" + std::to_string(i) + "]";
    std::vector<double> row = doubles(node[i], row_path);
    if (row.size() != width) {
      throw PolicyError(
              "'" + row_path + "' has " + std::to_string(row.size()) + " values, expected " +
              std::to_string(width));
    }
    out.push_back(std::move(row));
  }
  return out;
}

}  // namespace

void checkSupported(const PolicySpec & spec)
{
  if (spec.observation.layout != kGateRelativeLayout) {
    throw PolicyError(
            "observation layout '" + spec.observation.layout + "' is not supported, only '" +
            kGateRelativeLayout + "'");
  }
  if (spec.observation.action_history < 0) {
    throw PolicyError("observation.action_history must be >= 0");
  }
  const int expected = kGateRelativeSize + kActionSize * spec.observation.action_history;
  if (spec.observation.dim != expected) {
    throw PolicyError(
            "observation.dim is " + std::to_string(spec.observation.dim) + ", but " +
            std::to_string(kGateRelativeSize) + " + " + std::to_string(kActionSize) + " * " +
            std::to_string(spec.observation.action_history) + " = " + std::to_string(expected));
  }
  if (!(spec.observation.motor_speed_normalization > 0.0) ||
    !std::isfinite(spec.observation.motor_speed_normalization))
  {
    throw PolicyError("observation.motor_speed_normalization must be positive");
  }
  if (spec.action.contract != kBodyRatesContract) {
    throw PolicyError(
            "action contract '" + spec.action.contract + "' is not supported, only '" +
            kBodyRatesContract + "'");
  }
  if (!(spec.action.specific_thrust_max_ms2 > 0.0) ||
    !std::isfinite(spec.action.specific_thrust_max_ms2))
  {
    throw PolicyError("action.specific_thrust_max_ms2 must be positive");
  }
  for (const double rate : spec.action.max_body_rate_rad_s) {
    if (!(rate > 0.0) || !std::isfinite(rate)) {
      throw PolicyError("action.max_body_rate_rad_s must be positive");
    }
  }
  if (!(spec.dt > 0.0) || !std::isfinite(spec.dt)) {
    throw PolicyError("dt must be positive");
  }
}

MlpPolicy::MlpPolicy(PolicySpec spec, std::vector<Layer> layers)
: spec_(std::move(spec)), layers_(std::move(layers))
{
  checkSupported(spec_);
  if (layers_.empty()) {
    throw PolicyError("the network has no layers");
  }
  Eigen::Index inputs = spec_.observation.dim;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    const Layer & layer = layers_[i];
    if (layer.weight.cols() != inputs) {
      throw PolicyError(
              "layer " + std::to_string(i) + " takes " + std::to_string(layer.weight.cols()) +
              " inputs, the previous stage gives " + std::to_string(inputs));
    }
    if (layer.bias.size() != layer.weight.rows()) {
      throw PolicyError("layer " + std::to_string(i) + " bias does not match its weight");
    }
    if (!layer.weight.allFinite() || !layer.bias.allFinite()) {
      throw PolicyError("layer " + std::to_string(i) + " has non-finite parameters");
    }
    inputs = layer.weight.rows();
  }
  if (inputs != kActionSize) {
    throw PolicyError(
            "the last layer gives " + std::to_string(inputs) + " outputs, the action has " +
            std::to_string(kActionSize));
  }
}

std::shared_ptr<MlpPolicy> MlpPolicy::load(const std::string & path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::BadFile &) {
    throw PolicyError("cannot open policy file '" + path + "'");
  } catch (const YAML::Exception & e) {
    throw PolicyError("policy file '" + path + "' is not valid YAML: " + e.what());
  }

  try {
    expectString(root, "format", "", kPolicyFormat);
    const int version =
      scalar<int>(required(root, "format_version", ""), "format_version", "integer");
    if (version != kPolicyFormatVersion) {
      throw PolicyError(
              "format_version " + std::to_string(version) + " is not supported, only " +
              std::to_string(kPolicyFormatVersion));
    }

    PolicySpec spec;
    spec.file = path;
    spec.task = optionalString(root, "task", "race");
    spec.dt = positiveDouble(required(root, "dt", ""), "dt");

    const YAML::Node action = required(root, "action", "");
    spec.action.contract =
      scalar<std::string>(required(action, "contract", "action"), "action.contract", "string");
    spec.action.specific_thrust_max_ms2 = positiveDouble(
      required(action, "specific_thrust_max_ms2", "action"), "action.specific_thrust_max_ms2");
    const std::vector<double> rates = doubles(
      required(action, "max_body_rate_rad_s", "action"), "action.max_body_rate_rad_s");
    if (rates.size() != spec.action.max_body_rate_rad_s.size()) {
      throw PolicyError("'action.max_body_rate_rad_s' must hold 3 values");
    }
    std::copy(rates.begin(), rates.end(), spec.action.max_body_rate_rad_s.begin());

    const YAML::Node observation = required(root, "observation", "");
    spec.observation.layout = scalar<std::string>(
      required(observation, "layout", "observation"), "observation.layout", "string");
    spec.observation.action_history = scalar<int>(
      required(observation, "action_history", "observation"), "observation.action_history",
      "integer");
    spec.observation.dim =
      scalar<int>(required(observation, "dim", "observation"), "observation.dim", "integer");
    spec.observation.motor_speed_normalization = positiveDouble(
      required(observation, "motor_speed_normalization", "observation"),
      "observation.motor_speed_normalization");

    const YAML::Node network = required(root, "network", "");
    expectString(network, "activation", "network", kTanhActivation);
    expectString(network, "output", "network", kClipOutput);
    checkSupported(spec);

    const YAML::Node layers_node = required(network, "layers", "network");
    if (!layers_node.IsSequence() || layers_node.size() == 0) {
      throw PolicyError("'network.layers' must be a non-empty sequence");
    }
    std::vector<Layer> layers;
    for (std::size_t i = 0; i < layers_node.size(); ++i) {
      layers.push_back(parseLayer(layers_node[i], "network.layers[" + std::to_string(i) + "]"));
    }

    const YAML::Node source = root["source"];
    spec.source.checkpoint = optionalString(source, "checkpoint", "unknown");
    spec.source.checkpoint_sha256 = optionalString(source, "checkpoint_sha256", "unknown");
    spec.source.env_config = optionalString(source, "env_config", "unknown");
    spec.source.plant_config = optionalString(source, "plant_config", "unknown");
    spec.source.quadlab_commit = optionalString(source, "quadlab_commit", "unknown");
    const int dim = spec.observation.dim;
    auto policy = std::make_shared<MlpPolicy>(std::move(spec), std::move(layers));

    const YAML::Node golden = required(root, "golden", "");
    const double tolerance =
      finiteDouble(required(golden, "tolerance", "golden"), "golden.tolerance");
    if (tolerance < 0.0) {
      throw PolicyError("'golden.tolerance' must not be negative");
    }
    const auto golden_obs =
      rows(required(golden, "obs", "golden"), "golden.obs", static_cast<std::size_t>(dim));
    const auto golden_action =
      rows(required(golden, "action", "golden"), "golden.action", kActionSize);
    if (golden_obs.size() != golden_action.size()) {
      throw PolicyError(
              "'golden.obs' has " + std::to_string(golden_obs.size()) + " rows, 'golden.action' " +
              std::to_string(golden_action.size()));
    }
    if (golden_obs.empty()) {
      throw PolicyError("'golden' carries no rows, the network cannot be checked");
    }

    GoldenCheck check;
    check.rows = golden_obs.size();
    check.tolerance = tolerance;
    for (std::size_t row = 0; row < golden_obs.size(); ++row) {
      const Eigen::VectorXd obs = Eigen::Map<const Eigen::VectorXd>(
        golden_obs[row].data(), static_cast<Eigen::Index>(golden_obs[row].size()));
      const Action action_row = policy->act(obs);
      double error = 0.0;
      for (int k = 0; k < kActionSize; ++k) {
        error = std::max(error, std::abs(action_row[k] - golden_action[row][k]));
      }
      check.max_error = std::max(check.max_error, error);
      if (error > tolerance) {
        throw PolicyError(
                "golden row " + std::to_string(row) + " misses by " + number(error) +
                ", more than the tolerance " + number(tolerance) +
                ": the weights do not reproduce the exported network");
      }
    }
    policy->golden_check_ = check;
    return policy;
  } catch (const PolicyError & e) {
    throw PolicyError("policy file '" + path + "': " + e.what());
  }
}

Action MlpPolicy::act(const Eigen::VectorXd & observation) const
{
  if (observation.size() != spec_.observation.dim) {
    throw std::invalid_argument(
            "observation has " + std::to_string(observation.size()) +
            " entries, the policy reads " + std::to_string(spec_.observation.dim));
  }
  Eigen::VectorXd x = observation;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    x = layers_[i].weight * x + layers_[i].bias;
    if (i + 1 < layers_.size()) {
      x = x.unaryExpr([](double value) {return std::tanh(value);});
    }
  }
  Action action{};
  for (int k = 0; k < kActionSize; ++k) {
    action[k] = std::clamp(x(k), -1.0, 1.0);
  }
  return action;
}

PolicyBank PolicyBank::load(
  const std::vector<std::string> & names, const std::vector<std::string> & files)
{
  if (names.size() != files.size()) {
    throw PolicyError(
            "policies.names has " + std::to_string(names.size()) + " entries, policies.files " +
            std::to_string(files.size()));
  }
  if (names.empty()) {
    throw PolicyError("no policy is configured");
  }
  PolicyBank bank;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty()) {
      throw PolicyError("policies.names[" + std::to_string(i) + "] is empty");
    }
    if (files[i].empty()) {
      throw PolicyError(
              "policies.files[" + std::to_string(i) + "] for '" + names[i] +
              "' is empty: point it at an exported policy file");
    }
    if (bank.has(names[i])) {
      throw PolicyError("policy '" + names[i] + "' is configured twice");
    }
    try {
      bank.add(names[i], MlpPolicy::load(files[i]));
    } catch (const PolicyError & e) {
      throw PolicyError("policy '" + names[i] + "': " + e.what());
    }
  }
  return bank;
}

void PolicyBank::add(const std::string & name, std::shared_ptr<const Policy> policy)
{
  if (!policy) {
    throw PolicyError("policy '" + name + "' is null");
  }
  checkSupported(policy->spec());
  const double dt = policy->spec().dt;
  if (!policies_.empty() && std::abs(dt - dt_) > kDtTolerance) {
    throw PolicyError(
            "policy '" + name + "' steps at dt " + number(dt) + " s, the others at " +
            number(dt_) + " s: one controller runs one step");
  }
  if (policies_.empty()) {
    dt_ = dt;
  }
  policies_[name] = std::move(policy);
}

bool PolicyBank::has(const std::string & name) const
{
  return policies_.count(name) != 0;
}

std::shared_ptr<const Policy> PolicyBank::get(const std::string & name) const
{
  const auto it = policies_.find(name);
  if (it == policies_.end()) {
    throw PolicyError("no policy named '" + name + "'");
  }
  return it->second;
}

std::vector<std::string> PolicyBank::names() const
{
  std::vector<std::string> out;
  out.reserve(policies_.size());
  for (const auto & entry : policies_) {
    out.push_back(entry.first);
  }
  return out;
}

}  // namespace rl_policy
