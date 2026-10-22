#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "as2_rl_policy/core/angles.hpp"
#include "as2_rl_policy/core/race_pilot_course.hpp"

namespace
{

using rl_policy::CourseMatch;
using rl_policy::Gate;
using rl_policy::GateShape;
using rl_policy::SentCourse;
using rl_policy::Waypoint;
using rl_policy::WaypointId;

constexpr double kHalfPi = rl_policy::kPi / 2.0;
constexpr double kMatchM = 0.10;
constexpr double kMatchRad = 5.0 * rl_policy::kPi / 180.0;

Waypoint point(const std::string & id, double x, double y, double z, double yaw)
{
  return Waypoint{id, Eigen::Vector3d(x, y, z), yaw};
}

// yaw_angle travels as a float32.
double f32(double value)
{
  return static_cast<double>(static_cast<float>(value));
}

// fronton_s3_v4_low_lap as as2_race_pilot sent it in flight_07: 2 laps through the split.
std::vector<Waypoint> splitCourse()
{
  return {
    point("vehicle_ny", -0.0, 0.0, 0.1, 0.0),
    point("start_ny", 0.0, 0.0, 0.1, 0.0),
    point("gate01_gp_vn", 4.5, 0.0, 1.45, 0.0),
    point("gate02_ladderdown_gp_vn", 8.18, -9.0, 1.45, f32(-1.5708)),
    point("gate03_splitdown_gp_vn", 4.5, -18.0, 1.45, f32(-1.5708)),
    point("gate03_splitup_v_gp_vn", 4.5, -18.0, 4.15, f32(1.5708)),
    point("gate04_gp_vn", 0.82, -9.0, 1.45, f32(1.5708)),
    point("gate01_gp_vn", 4.5, 0.0, 1.45, 0.0),
    point("gate02_ladderdown_lap2_gp_vn", 8.18, -9.0, 1.45, f32(-1.5708)),
    point("gate03_splitdown_lap2_gp_vn", 4.5, -18.0, 1.45, f32(-1.5708)),
    point("gate03_splitup_v_lap2_gp_vn", 4.5, -18.0, 4.15, f32(1.5708)),
    point("gate04_lap2_gp_vn", 0.82, -9.0, 1.45, f32(1.5708)),
    point("gate01_lap2_gp_vn", 4.5, 0.0, 1.45, 0.0),
    point("end", 8.18, -5.0, 1.45, 0.0),
  };
}

// fronton_s3_v3 the same way: the far end's turn is the virtual gate03_fake_v, never faced.
std::vector<Waypoint> frontonCourse()
{
  return {
    point("vehicle_ny", 0.0, 0.0, 0.1, 0.0),
    point("start_ny", 0.0, 0.0, 0.1, 0.0),
    point("gate01_gp_vn", 4.5, 0.0, 1.45, 0.0),
    point("gate02_ladderdown_gp_vn", 8.18, -9.0, 1.45, f32(-1.5708)),
    point("gate03_fake_v_gp_ny_vn", 4.5, -14.0, 1.45, f32(3.14159)),
    point("gate04_gp_vn", 0.82, -9.0, 1.45, f32(1.5708)),
    point("gate01_gp_vn", 4.5, 0.0, 1.45, 0.0),
    point("gate02_ladderdown_lap2_gp_vn", 8.18, -9.0, 1.45, f32(-1.5708)),
    point("gate03_fake_v_lap2_gp_ny_vn", 4.5, -14.0, 1.45, f32(3.14159)),
    point("gate04_lap2_gp_vn", 0.82, -9.0, 1.45, f32(1.5708)),
    point("gate01_lap2_gp_vn", 4.5, 0.0, 1.45, 0.0),
    point("end", 8.18, -5.0, 1.45, 0.0),
  };
}

// The circuit FRONTON_v9mcsrlt trained on: rl/fronton/rl_policy.yaml's course.gates_*.
std::vector<Gate> trainedFronton()
{
  const double x[] = {4.5, 8.18, 4.5, 0.82};
  const double y[] = {0.0, -9.0, -14.0, -9.0};
  const double yaw[] = {0.0, -1.5708, 3.14159, 1.5708};
  std::vector<Gate> gates(4);
  for (std::size_t i = 0; i < gates.size(); ++i) {
    gates[i].position = Eigen::Vector3d(x[i], y[i], 1.45);
    gates[i].yaw = yaw[i];
    gates[i].is_virtual = i == 2;
  }
  return gates;
}

CourseMatch matchFronton(const std::vector<Waypoint> & waypoints)
{
  const SentCourse course = rl_policy::sentCourse(waypoints);
  return rl_policy::matchCourse(
    course.lap(), course.lapNames(), trainedFronton(), kMatchM, kMatchRad);
}

}  // namespace

TEST(WaypointId, StripsTheMarksAndTheLapTag) {
  const WaypointId gate = rl_policy::parseWaypointId("gate02_ladderdown_lap2_gp_vn");
  EXPECT_EQ(gate.gate, "gate02_ladderdown");
  EXPECT_TRUE(gate.crossing);
  const WaypointId every = rl_policy::parseWaypointId("gate01_gp_ny_ls_sy_vn");
  EXPECT_EQ(every.gate, "gate01");
  EXPECT_TRUE(every.crossing);
  const WaypointId virtual_gate = rl_policy::parseWaypointId("gate03_splitup_v_gp_vn");
  EXPECT_EQ(virtual_gate.gate, "gate03_splitup_v");
  EXPECT_TRUE(virtual_gate.crossing);
  // Not crossings: the vehicle, the start, a pass-through point and the end.
  EXPECT_FALSE(rl_policy::parseWaypointId("vehicle_ny").crossing);
  EXPECT_FALSE(rl_policy::parseWaypointId("start_ny").crossing);
  EXPECT_FALSE(rl_policy::parseWaypointId("gate02_ny_rt").crossing);
  EXPECT_FALSE(rl_policy::parseWaypointId("end").crossing);
  EXPECT_EQ(rl_policy::parseWaypointId("end").gate, "end");
  // A label that is only a mark keeps it as its name.
  EXPECT_EQ(rl_policy::parseWaypointId("gp").gate, "gp");
}

TEST(WaypointId, VirtualAndOctagonNamesFollowTheStack) {
  EXPECT_TRUE(rl_policy::isVirtualGateName("gate03_fake_v"));
  EXPECT_TRUE(rl_policy::isVirtualGateName("gate01_r_v"));
  EXPECT_TRUE(rl_policy::isVirtualGateName("gatePow_v"));
  EXPECT_FALSE(rl_policy::isVirtualGateName("gate02_ladderdown"));
  EXPECT_FALSE(rl_policy::isVirtualGateName("gate_aux01"));
  EXPECT_TRUE(rl_policy::isOctagonGateName("gate03_oct"));
  EXPECT_FALSE(rl_policy::isOctagonGateName("gate03_octagonal"));
}

TEST(SentCourse, TheSplitCourseFlightSevenFlew) {
  const SentCourse course = rl_policy::sentCourse(splitCourse());
  EXPECT_EQ(course.crossings.size(), 11u);
  EXPECT_EQ(course.lap_size, 5u);
  EXPECT_EQ(course.laps, 2);
  EXPECT_EQ(course.extra, 1u);
  EXPECT_EQ(course.ignored, 3u);
  const std::vector<std::string> names = course.lapNames();
  const std::vector<std::string> expected = {
    "gate01", "gate02_ladderdown", "gate03_splitdown", "gate03_splitup_v", "gate04"};
  EXPECT_EQ(names, expected);
  const std::vector<Gate> lap = course.lap();
  for (std::size_t i = 0; i < lap.size(); ++i) {
    EXPECT_EQ(lap[i].is_virtual, i == 3) << names[i];
    EXPECT_EQ(lap[i].shape, GateShape::Square) << names[i];
  }
  EXPECT_NEAR(lap[3].position.z(), 4.15, 1e-12);
  EXPECT_NEAR(lap[3].yaw, kHalfPi, 1e-5);
}

TEST(SentCourse, TheFrontonCourseIsTwoLapsOfFourAndItsClosingGate) {
  const SentCourse course = rl_policy::sentCourse(frontonCourse());
  EXPECT_EQ(course.crossings.size(), 9u);
  EXPECT_EQ(course.lap_size, 4u);
  EXPECT_EQ(course.laps, 2);
  EXPECT_EQ(course.extra, 1u);
  EXPECT_EQ(course.ignored, 3u);
  EXPECT_EQ(course.names.back(), "gate01");
}

TEST(SentCourse, RefusesACourseWithoutACrossingOrWithANonFiniteOne) {
  EXPECT_THROW(
    rl_policy::sentCourse({point("vehicle_ny", 0, 0, 0, 0), point("end", 1, 1, 1, 0)}),
    std::invalid_argument);
  std::vector<Waypoint> bad = frontonCourse();
  bad[3].position.y() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(rl_policy::sentCourse(bad), std::invalid_argument);
}

TEST(MatchCourse, TheFrontonCourseMatchesTheCircuitItsNetworkTrainedOn) {
  const CourseMatch match = matchFronton(frontonCourse());
  EXPECT_TRUE(match.matches) << match.reason;
  EXPECT_LT(match.max_position_error_m, 1e-9);
  EXPECT_LT(match.max_yaw_error_rad, 1e-5);
}

TEST(MatchCourse, TheSplitIsNotTheCircuitTheFrontonNetworkTrainedOn) {
  const CourseMatch match = matchFronton(splitCourse());
  EXPECT_FALSE(match.matches);
  EXPECT_NE(match.reason.find("5 gates"), std::string::npos) << match.reason;
  EXPECT_NE(match.reason.find("gate03_splitdown"), std::string::npos) << match.reason;
}

// Every crossing of the gate, in every lap, changed alike.
std::vector<Waypoint> changed(
  const std::string & gate, const Eigen::Vector3d & shift, double yaw = 0.0, bool set_yaw = false)
{
  std::vector<Waypoint> course = frontonCourse();
  for (Waypoint & waypoint : course) {
    if (rl_policy::parseWaypointId(waypoint.id).gate == gate) {
      waypoint.position += shift;
      waypoint.yaw = set_yaw ? yaw : waypoint.yaw;
    }
  }
  return course;
}

TEST(MatchCourse, AGateMovedPastTheToleranceIsRefused) {
  EXPECT_TRUE(matchFronton(changed("gate02_ladderdown", Eigen::Vector3d(0.05, 0.0, 0.0))).matches);
  const CourseMatch match =
    matchFronton(changed("gate02_ladderdown", Eigen::Vector3d(0.2, 0.0, 0.0)));
  EXPECT_FALSE(match.matches);
  EXPECT_NE(match.reason.find("gate 1 (gate02_ladderdown)"), std::string::npos) << match.reason;
}

TEST(MatchCourse, AGateCrossedTheOtherWayIsRefused) {
  const CourseMatch match =
    matchFronton(changed("gate04", Eigen::Vector3d::Zero(), -kHalfPi, true));
  EXPECT_FALSE(match.matches);
  EXPECT_NE(match.reason.find("gate 3 (gate04)"), std::string::npos) << match.reason;
}

TEST(MatchCourse, ALapThatDiffersFromTheNextIsNoLapOfTheTrainedCircuit) {
  std::vector<Waypoint> uneven = frontonCourse();
  uneven[3].position.x() += 0.05;
  const SentCourse course = rl_policy::sentCourse(uneven);
  EXPECT_NE(course.lap_size, 4u);
  EXPECT_FALSE(matchFronton(uneven).matches);
}

TEST(MatchCourse, AVirtualGateThatStandsIsRefused) {
  std::vector<Waypoint> standing = frontonCourse();
  for (Waypoint & waypoint : standing) {
    const std::size_t at = waypoint.id.find("gate03_fake_v");
    if (at != std::string::npos) {
      waypoint.id.replace(at, std::string("gate03_fake_v").size(), "gate03_fake");
    }
  }
  const CourseMatch match = matchFronton(standing);
  EXPECT_FALSE(match.matches);
  EXPECT_NE(match.reason.find("not virtual"), std::string::npos) << match.reason;
}
