#include "as2_rl_policy/core/course.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "as2_rl_policy/core/angles.hpp"

namespace rl_policy
{

bool Bounds::contains(double x, double y, double margin) const
{
  return x >= x_min - margin && x <= x_max + margin && y >= y_min - margin &&
         y <= y_max + margin;
}

Eigen::Vector2d Bounds::clamp(double x, double y) const
{
  return Eigen::Vector2d(std::clamp(x, x_min, x_max), std::clamp(y, y_min, y_max));
}

Course::Course(std::vector<Gate> gates, Bounds bounds)
: gates_(std::move(gates)), bounds_(bounds)
{
  if (gates_.empty()) {
    throw std::invalid_argument("a course needs at least one gate");
  }
  if (!(bounds_.x_min <= bounds_.x_max) || !(bounds_.y_min <= bounds_.y_max)) {
    throw std::invalid_argument("course bounds must be ordered [min, max]");
  }
  const std::size_t n = gates_.size();
  relative_positions_.assign(n, Eigen::Vector3d::Zero());
  relative_yaws_.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t prev = (i + n - 1) % n;
    const Eigen::Vector3d delta = gates_[i].position - gates_[prev].position;
    const double c = std::cos(gates_[prev].yaw);
    const double s = std::sin(gates_[prev].yaw);
    relative_positions_[i] =
      Eigen::Vector3d(c * delta.x() + s * delta.y(), -s * delta.x() + c * delta.y(), delta.z());
    relative_yaws_[i] = wrapPi(gates_[i].yaw - gates_[prev].yaw);
  }
}

Course Course::single(const Gate & gate)
{
  return Course(std::vector<Gate>{gate});
}

const Eigen::Vector3d & Course::relativePosition(std::size_t index) const
{
  return relative_positions_[index % gates_.size()];
}

double Course::relativeYaw(std::size_t index) const
{
  return relative_yaws_[index % gates_.size()];
}

Eigen::Vector4d Course::nextBlock(std::size_t target) const
{
  const std::size_t m = next(target);
  const Eigen::Vector3d & position = relative_positions_[m];
  return Eigen::Vector4d(position.x(), position.y(), position.z(), relative_yaws_[m]);
}

}  // namespace rl_policy
