#ifndef AS2_RL_POLICY__CORE__COURSE_HPP_
#define AS2_RL_POLICY__CORE__COURSE_HPP_

#include <Eigen/Dense>

#include <cstddef>
#include <limits>
#include <vector>

namespace rl_policy
{

struct Gate
{
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double yaw = 0.0;
};

struct Bounds
{
  double x_min = -std::numeric_limits<double>::infinity();
  double x_max = std::numeric_limits<double>::infinity();
  double y_min = -std::numeric_limits<double>::infinity();
  double y_max = std::numeric_limits<double>::infinity();

  bool contains(double x, double y, double margin = 0.0) const;

  Eigen::Vector2d clamp(double x, double y) const;
};

// A looped course: the gate after the last is the first, as the kernel's loop tables are.
class Course
{
public:
  explicit Course(std::vector<Gate> gates, Bounds bounds = Bounds());

  static Course single(const Gate & gate);

  std::size_t size() const {return gates_.size();}

  const std::vector<Gate> & gates() const {return gates_;}

  const Gate & gate(std::size_t index) const {return gates_[index % gates_.size()];}

  std::size_t next(std::size_t index) const {return (index + 1) % gates_.size();}

  // Gate i relative to gate i - 1: Rz(-yaw_{i-1}) (g_i - g_{i-1}), z not rotated.
  const Eigen::Vector3d & relativePosition(std::size_t index) const;

  // wrapPi(yaw_i - yaw_{i-1}).
  double relativeYaw(std::size_t index) const;

  // Observation entries 16..19 while flying at the target: the next gate's relative pose.
  Eigen::Vector4d nextBlock(std::size_t target) const;

  const Bounds & bounds() const {return bounds_;}

private:
  std::vector<Gate> gates_;
  std::vector<Eigen::Vector3d> relative_positions_;
  std::vector<double> relative_yaws_;
  Bounds bounds_;
};

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__COURSE_HPP_
