#ifndef AS2_RL_POLICY_PLUGIN__TESTS__FIXTURE_UTILS_HPP_
#define AS2_RL_POLICY_PLUGIN__TESTS__FIXTURE_UTILS_HPP_

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "as2_rl_policy/core/course.hpp"
#include "as2_rl_policy/core/observation.hpp"

namespace rl_policy_test
{

inline std::string fixturePath(const std::string & name)
{
  return std::string(RL_POLICY_FIXTURES_DIR) + "/" + name;
}

inline YAML::Node loadFixture(const std::string & name)
{
  return YAML::LoadFile(fixturePath(name));
}

inline std::vector<double> doubles(const YAML::Node & node)
{
  return node.as<std::vector<double>>();
}

inline std::vector<rl_policy::Gate> courseGates(const YAML::Node & course)
{
  const auto x = doubles(course["gates_x"]);
  const auto y = doubles(course["gates_y"]);
  const auto z = doubles(course["gates_z"]);
  const auto yaw = doubles(course["gates_yaw"]);
  std::vector<rl_policy::Gate> gates(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    gates[i].position = Eigen::Vector3d(x[i], y[i], z[i]);
    gates[i].yaw = yaw[i];
  }
  return gates;
}

// A fixture state row: [x y z, vx vy vz, qw qx qy qz, p q r, W1..W4].
inline rl_policy::VehicleState stateFromRow(const std::vector<double> & row)
{
  rl_policy::VehicleState state;
  state.position = Eigen::Vector3d(row[0], row[1], row[2]);
  state.velocity = Eigen::Vector3d(row[3], row[4], row[5]);
  state.orientation = Eigen::Quaterniond(row[6], row[7], row[8], row[9]);
  state.body_rates = Eigen::Vector3d(row[10], row[11], row[12]);
  for (std::size_t motor = 0; motor < 4; ++motor) {
    state.motor_speeds[motor] = row[13 + motor];
  }
  return state;
}

// A fresh file under the system temporary directory.
inline std::string temporaryFile(const std::string & stem)
{
  static std::atomic<int> counter{0};
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
    ("as2_rl_policy_" + stem + "_" + std::to_string(stamp) + "_" + std::to_string(counter++) +
    ".yaml");
  return path.string();
}

inline std::string writeYaml(const YAML::Node & node, const std::string & stem)
{
  const std::string path = temporaryFile(stem);
  YAML::Emitter emitter;
  emitter.SetDoublePrecision(17);
  emitter << node;
  std::ofstream(path) << emitter.c_str() << "\n";
  return path;
}

}  // namespace rl_policy_test

#endif  // AS2_RL_POLICY_PLUGIN__TESTS__FIXTURE_UTILS_HPP_
