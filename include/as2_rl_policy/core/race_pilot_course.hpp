#ifndef AS2_RL_POLICY__CORE__RACE_PILOT_COURSE_HPP_
#define AS2_RL_POLICY__CORE__RACE_PILOT_COURSE_HPP_

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <vector>

#include "as2_rl_policy/core/course.hpp"

namespace rl_policy
{

// One point of the course as2_race_pilot sends: its id, position and yaw_angle.
struct Waypoint
{
  std::string id;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double yaw = 0.0;

  bool operator==(const Waypoint & other) const
  {
    return id == other.id && position == other.position && yaw == other.yaw;
  }
};

// A waypoint id as as2_race_pilot writes it: <label>[_lapN][_gp][_ny][_rt|_ls][_sy][_vn].
struct WaypointId
{
  // The label without its lap tag and marks: the gate's name for a crossing.
  std::string gate;
  // The _gp mark: a crossing of the gate, the only kind of point the race reads.
  bool crossing = false;
};

WaypointId parseWaypointId(const std::string & id);

// The stack's rule: a gate whose name contains "_v" is virtual.
bool isVirtualGateName(const std::string & name);

// A gate whose name ends in "_oct" is an octagon.
bool isOctagonGateName(const std::string & name);

// The crossings of a sent course in order, and the laps they make.
struct SentCourse
{
  std::vector<Gate> crossings;
  std::vector<std::string> names;
  // The smallest period of the crossing list: one lap's gates.
  std::size_t lap_size = 0;
  int laps = 0;
  // Crossings past the last full lap: the closing first gate of a closed circuit.
  std::size_t extra = 0;
  // The points that are no crossing: the vehicle, the start, pass-through points, the end.
  std::size_t ignored = 0;

  std::vector<Gate> lap() const;

  std::vector<std::string> lapNames() const;
};

// Every _gp point as a gate, the MPCC's other marks and points dropped. Throws
// std::invalid_argument when there is no crossing or one is not finite.
SentCourse sentCourse(const std::vector<Waypoint> & waypoints);

struct CourseMatch
{
  bool matches = false;
  // Why it does not, naming the first gate that differs.
  std::string reason;
  double max_position_error_m = 0.0;
  double max_yaw_error_rad = 0.0;
};

// One lap of a sent course against the trained one: count and order, position, yaw, virtual
// flag and shape, gate by gate from the first.
CourseMatch matchCourse(
  const std::vector<Gate> & lap, const std::vector<std::string> & names,
  const std::vector<Gate> & trained, double position_tolerance_m, double yaw_tolerance_rad);

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__RACE_PILOT_COURSE_HPP_
