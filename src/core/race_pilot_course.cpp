#include "as2_rl_policy/core/race_pilot_course.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "as2_rl_policy/core/angles.hpp"
#include "as2_rl_policy/core/gates.hpp"

namespace rl_policy
{

namespace
{

// The marks as2_race_pilot appends to a waypoint's label (race_pilot_node.cpp, buildMission).
bool isMark(const std::string & field)
{
  return field == "gp" || field == "ny" || field == "rt" || field == "ls" || field == "sy" ||
         field == "vn";
}

// The lap tag expandMission() gives the second lap's crossings on: lap2, lap3, ...
bool isLapTag(const std::string & field)
{
  return field.size() > 3 && field.compare(0, 3, "lap") == 0 &&
         std::all_of(
    field.begin() + 3, field.end(),
    [](unsigned char c) {return std::isdigit(c) != 0;});
}

bool sameCrossing(
  const Gate & a, const std::string & name_a, const Gate & b, const std::string & name_b)
{
  return name_a == name_b && (a.position - b.position).norm() < 1.0e-6 &&
         std::abs(wrapPi(a.yaw - b.yaw)) < 1.0e-6;
}

std::string format(const char * pattern, double a, double b, double c)
{
  char text[96];
  std::snprintf(text, sizeof(text), pattern, a, b, c);
  return text;
}

}  // namespace

WaypointId parseWaypointId(const std::string & id)
{
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = id.find('_', begin);
    fields.push_back(id.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }

  WaypointId parsed;
  while (fields.size() > 1 && isMark(fields.back())) {
    parsed.crossing = parsed.crossing || fields.back() == "gp";
    fields.pop_back();
  }
  if (fields.size() > 1 && isLapTag(fields.back())) {
    fields.pop_back();
  }
  for (std::size_t i = 0; i < fields.size(); ++i) {
    parsed.gate += (i == 0 ? "" : "_") + fields[i];
  }
  return parsed;
}

bool isVirtualGateName(const std::string & name)
{
  return name.find("_v") != std::string::npos;
}

bool isOctagonGateName(const std::string & name)
{
  const std::string suffix = "_oct";
  return name.size() >= suffix.size() &&
         name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<Gate> SentCourse::lap() const
{
  return std::vector<Gate>(crossings.begin(), crossings.begin() + lap_size);
}

std::vector<std::string> SentCourse::lapNames() const
{
  return std::vector<std::string>(names.begin(), names.begin() + lap_size);
}

SentCourse sentCourse(const std::vector<Waypoint> & waypoints)
{
  SentCourse course;
  for (const Waypoint & waypoint : waypoints) {
    const WaypointId id = parseWaypointId(waypoint.id);
    if (!id.crossing) {
      ++course.ignored;
      // expandMission() writes the track's end_point last, as "end".
      if (&waypoint == &waypoints.back() && id.gate == kEndPointId &&
        waypoint.position.allFinite())
      {
        course.end_point = waypoint.position;
      }
      continue;
    }
    if (!waypoint.position.allFinite() || !std::isfinite(waypoint.yaw)) {
      throw std::invalid_argument("the crossing '" + waypoint.id + "' is not finite");
    }
    Gate gate;
    gate.position = waypoint.position;
    gate.yaw = waypoint.yaw;
    gate.is_virtual = isVirtualGateName(id.gate);
    gate.shape = isOctagonGateName(id.gate) ? GateShape::Octagon : GateShape::Square;
    course.crossings.push_back(gate);
    course.names.push_back(id.gate);
  }
  const std::size_t n = course.crossings.size();
  if (n == 0) {
    throw std::invalid_argument(
            "the course has no crossing: no waypoint id carries the _gp mark");
  }
  for (std::size_t period = 1; period <= n; ++period) {
    bool periodic = true;
    for (std::size_t k = period; k < n && periodic; ++k) {
      periodic = sameCrossing(
        course.crossings[k], course.names[k], course.crossings[k % period],
        course.names[k % period]);
    }
    if (periodic) {
      course.lap_size = period;
      break;
    }
  }
  course.laps = static_cast<int>(n / course.lap_size);
  course.extra = n % course.lap_size;
  return course;
}

CourseMatch matchCourse(
  const std::vector<Gate> & lap, const std::vector<std::string> & names,
  const std::vector<Gate> & trained, double position_tolerance_m, double yaw_tolerance_rad)
{
  CourseMatch match;
  const auto name = [&names](std::size_t i) {
      return i < names.size() ? " (" + names[i] + ")" : std::string();
    };
  if (lap.size() != trained.size()) {
    std::string sent;
    for (std::size_t i = 0; i < names.size(); ++i) {
      sent += (i == 0 ? "" : ", ") + names[i];
    }
    match.reason = "a lap of the sent course has " + std::to_string(lap.size()) + " gates (" +
      sent + "), the race policy trained on " + std::to_string(trained.size());
    return match;
  }
  for (std::size_t i = 0; i < lap.size(); ++i) {
    const double position_error = (lap[i].position - trained[i].position).norm();
    const double yaw_error = std::abs(wrapPi(lap[i].yaw - trained[i].yaw));
    match.max_position_error_m = std::max(match.max_position_error_m, position_error);
    match.max_yaw_error_rad = std::max(match.max_yaw_error_rad, yaw_error);
    const std::string gate = "gate " + std::to_string(i) + name(i);
    if (position_error > position_tolerance_m) {
      char text[64];
      std::snprintf(
        text, sizeof(text), ": %.3f m apart, the tolerance is %.3f m", position_error,
        position_tolerance_m);
      match.reason = gate + " is at " +
        format("[%.3f, %.3f, %.3f]", lap[i].position.x(), lap[i].position.y(), lap[i].position.z()) +
        ", the trained one at " +
        format(
        "[%.3f, %.3f, %.3f]", trained[i].position.x(), trained[i].position.y(),
        trained[i].position.z()) + text;
      return match;
    }
    if (yaw_error > yaw_tolerance_rad) {
      char text[160];
      std::snprintf(
        text, sizeof(text), " is crossed at yaw %.3f rad, the trained one at %.3f rad: %.1f deg "
        "apart, the tolerance is %.1f deg", lap[i].yaw, trained[i].yaw, yaw_error * 180.0 / kPi,
        yaw_tolerance_rad * 180.0 / kPi);
      match.reason = gate + text;
      return match;
    }
    if (lap[i].is_virtual != trained[i].is_virtual) {
      match.reason = gate + (lap[i].is_virtual ? " is virtual, the trained one is not" :
        " is not virtual, the trained one is");
      return match;
    }
    if (lap[i].shape != trained[i].shape) {
      match.reason = gate + " is " + toString(lap[i].shape) + ", the trained one " +
        toString(trained[i].shape);
      return match;
    }
  }
  match.matches = true;
  return match;
}

}  // namespace rl_policy
