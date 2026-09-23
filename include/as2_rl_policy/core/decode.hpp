#ifndef AS2_RL_POLICY__CORE__DECODE_HPP_
#define AS2_RL_POLICY__CORE__DECODE_HPP_

#include <Eigen/Dense>

#include "as2_rl_policy/core/policy.hpp"

namespace rl_policy
{

struct Command
{
  double thrust_n = 0.0;
  // Body frame (FLU), rad/s.
  Eigen::Vector3d rates = Eigen::Vector3d::Zero();
};

Action clip(const Action & action);

// The kernel's decodeNormalizedAction, re-dimensionalized with the platform mass.
Command decode(const Action & action, const ActionSpec & spec, double mass_kg);

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__DECODE_HPP_
