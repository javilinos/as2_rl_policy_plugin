#ifndef AS2_RL_POLICY__CORE__ANGLES_HPP_
#define AS2_RL_POLICY__CORE__ANGLES_HPP_

#include <Eigen/Geometry>

namespace rl_policy
{

inline constexpr double kPi = 3.14159265358979323846;

// The kernel's wrapPi (multirotor_cpp_rl types.hpp), range [-pi, pi).
double wrapPi(double angle);

// The kernel's quaternionToEuler (multirotor_cpp state.hpp); it normalizes the quaternion.
void quaternionToEuler(
  const Eigen::Quaterniond & quaternion, double & roll, double & pitch, double & yaw);

double yawFromQuaternion(const Eigen::Quaterniond & quaternion);

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__ANGLES_HPP_
