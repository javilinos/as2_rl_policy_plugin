#include "as2_rl_policy/core/angles.hpp"

#include <cmath>

namespace rl_policy
{

double wrapPi(double angle)
{
  angle = std::fmod(angle + kPi, 2.0 * kPi);
  if (angle < 0) {
    angle += 2.0 * kPi;
  }
  return angle - kPi;
}

void quaternionToEuler(
  const Eigen::Quaterniond & _quaternion, double & roll, double & pitch, double & yaw)
{
  Eigen::Quaterniond quaternion = _quaternion.normalized();

  double sinr = 2.0 * (quaternion.w() * quaternion.x() + quaternion.y() * quaternion.z());
  double cosr = 1.0 - 2.0 * (quaternion.x() * quaternion.x() + quaternion.y() * quaternion.y());
  roll = std::atan2(sinr, cosr);

  double sinp = 2.0 * (quaternion.w() * quaternion.y() - quaternion.z() * quaternion.x());
  if (std::abs(sinp) >= 1.0) {
    pitch = std::copysign(M_PI / 2, sinp);
  } else {
    pitch = std::asin(sinp);
  }

  double siny = 2.0 * (quaternion.w() * quaternion.z() + quaternion.x() * quaternion.y());
  double cosy = 1.0 - 2.0 * (quaternion.y() * quaternion.y() + quaternion.z() * quaternion.z());
  yaw = std::atan2(siny, cosy);
}

double yawFromQuaternion(const Eigen::Quaterniond & quaternion)
{
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  quaternionToEuler(quaternion, roll, pitch, yaw);
  return yaw;
}

}  // namespace rl_policy
