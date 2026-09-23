#include "as2_rl_policy/core/decode.hpp"

#include <algorithm>

namespace rl_policy
{

Action clip(const Action & action)
{
  Action out{};
  for (int k = 0; k < kActionSize; ++k) {
    out[k] = std::clamp(action[k], -1.0, 1.0);
  }
  return out;
}

Command decode(const Action & action, const ActionSpec & spec, double mass_kg)
{
  const Action a = clip(action);
  Command command;
  command.thrust_n = 0.5 * (a[0] + 1.0) * spec.specific_thrust_max_ms2 * mass_kg;
  for (int axis = 0; axis < 3; ++axis) {
    command.rates(axis) = a[axis + 1] * spec.max_body_rate_rad_s[axis];
  }
  return command;
}

}  // namespace rl_policy
