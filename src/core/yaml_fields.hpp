#ifndef AS2_RL_POLICY__CORE__YAML_FIELDS_HPP_
#define AS2_RL_POLICY__CORE__YAML_FIELDS_HPP_

// What the readers of quadlab's files share: required keys, typed values and network layers, each
// refused as a PolicyError with its path. Private to src/core.

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "as2_rl_policy/core/policy.hpp"

namespace rl_policy
{

inline std::string number(double value)
{
  std::ostringstream stream;
  stream.precision(9);
  stream << value;
  return stream.str();
}

inline std::string join(const std::string & path, const std::string & key)
{
  return path.empty() ? key : path + "." + key;
}

inline YAML::Node required(
  const YAML::Node & parent, const std::string & key, const std::string & path)
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

inline double finiteDouble(const YAML::Node & node, const std::string & path)
{
  const double value = scalar<double>(node, path, "number");
  if (!std::isfinite(value)) {
    throw PolicyError("'" + path + "' is not finite");
  }
  return value;
}

inline double positiveDouble(const YAML::Node & node, const std::string & path)
{
  const double value = finiteDouble(node, path);
  if (!(value > 0.0)) {
    throw PolicyError("'" + path + "' must be positive, got " + number(value));
  }
  return value;
}

inline std::vector<double> doubles(const YAML::Node & node, const std::string & path)
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

inline std::string optionalString(
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

inline void expectString(
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
inline double float32(double value)
{
  return static_cast<double>(static_cast<float>(value));
}

inline MlpPolicy::Layer parseLayer(const YAML::Node & node, const std::string & path)
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

inline std::vector<std::vector<double>> rows(
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

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__YAML_FIELDS_HPP_
