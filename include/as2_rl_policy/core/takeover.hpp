#ifndef AS2_RL_POLICY__CORE__TAKEOVER_HPP_
#define AS2_RL_POLICY__CORE__TAKEOVER_HPP_

// Two networks around the moving bars, on top of the race policy they trained with (main), which
// keeps flying the lap (quadlab.envs.takeover). In a window around a barred gate, with g its gain:
//
//   row = main's row + main's action + the readings + the bar's type + time to the gate + g
//   before the plane: main's row[1:2] -= aim_m * g * aim(row), and main's action is main's on that
//   row[31:34] = main's action
//   action = clip(main's action + authority * g * residual(row), -1, 1)
//
// The readings are the detector's, of the next obstacle only, and the networks see nothing else of
// the bar: not its speed, not where it will be.
//
// A recurrent aim (quadlab.envs.takeover.MemoryAim) reads its columns of the row on every tick from
// the gate before its bar to the hand-back, gain or none, and forgets outside that. With show_aim
// the residual's row also holds g * aim, zero past the plane, in two columns before g.

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "as2_rl_policy/core/angles.hpp"
#include "as2_rl_policy/core/course.hpp"
#include "as2_rl_policy/core/observation.hpp"
#include "as2_rl_policy/core/policy.hpp"

namespace rl_policy
{

inline constexpr const char kTakeoverFormat[] = "quadlab_takeover";
// 2 adds reading.frame_hz: a reader that skips it thins the frames its networks expect whole, so it must refuse the file.
// 3 adds the recurrent aim (aim.memory) and show_aim, which a reader before it would fly as neither.
// 4 adds the aim on a rule (aim.rule), which this reader refuses: its own reader flies it.
// 5 adds window.handback_next_m, the hand-back past the gate after the bar, which a reader before it would make early.
// 6 adds window.lane_hold_s, the rotating bar's lane kept near its plane, which a reader before it would fly without.
// 7 adds window.slide_commit_s and slide_inset_m with the sliding bars' slide block, a sliding bar's lane committed near
// its plane, which a reader before it would fly without.
// 8 adds window.turn_floor_inset_m with the rotating bars' turn block, the rotating bar's held lane at least as deep as
// its middle, which a reader before it would fly without.
inline constexpr int kTakeoverFormatVersion = 8;
// main's row: the R6 block and two commands.
inline constexpr int kMainRowSize = 31;
inline constexpr int kAimSize = 2;
// Two values and the age.
inline constexpr int kReadingColumns = 3;
inline constexpr int kBarKinds = 3;
// A rotating bar at a detector's position of 1, detect_gates' barPlacement.
inline constexpr double kBarTurnRad = 179.0 * kPi / 180.0;

// The order of the row's bar type columns.
enum class BarKind
{
  UpDown = 0,
  LeftRight = 1,
  Rotating = 2,
  // No bar stands in that gate: its window never opens and the race policy flies it alone.
  None = 3,
};

// The side of the approach a detector's position of 1 puts a left-right bar on. The networks
// trained on the right; obstacle_gates reports the gate's left.
enum class BarSide
{
  Left,
  Right,
};

// quadlab.envs.takeover.TakeoverVecEnv's four modes.
enum class TakeoverMode
{
  Before = 0,
  Active = 1,
  HandingBack = 2,
  Tail = 3,
};

const char * toString(BarKind kind);

const char * toString(BarSide side);

const char * toString(TakeoverMode mode);

// The word of a mission's obstacles: throws std::invalid_argument for anything but "vertical",
// "horizontal" or "rotating", or "none" for a gate without its bar.
BarKind barKindFromString(const std::string & name);

// Throws std::invalid_argument for anything but "left" or "right".
BarSide barSideFromString(const std::string & name);

// The row's two values of a detector's position (0 to 1): a sliding bar's place over its travel,
// up and to the approach's right, then zero; a rotating bar's sine and cosine of twice its angle,
// from the gate's left towards its up.
std::array<double, 2> barValues(BarKind kind, double position, BarSide one_at);

struct TakeoverGate
{
  std::string name;
  // In the course.
  std::size_t index = 0;
  // Positive: the window opens this long before the plane, while the gate before is the target.
  double open_s = 0.0;
  // The bars the networks trained on here.
  std::vector<BarKind> kinds;
};

// Where a commit at the sliding bars draws its lanes (format 7's slide): a line through the readings of
// the last window_s, `minimum` of them at least, and each barred gate's opening and sliding bars, by
// the file's gates.
struct SlideGeometry
{
  struct Bar
  {
    double travel_m = 0.0;
    double half_width_m = 0.0;
  };
  struct Gate
  {
    bool present = false;
    double opening_half_m = 0.0;
    // An up-down bar's, a left-right bar's.
    std::array<std::optional<Bar>, 2> bars;
  };
  double window_s = 0.0;
  int minimum = 0;
  std::vector<Gate> gates;
};

// Where a floor at the rotating bars draws its lanes (format 8's turn): a line through the angle readings of
// the last window_s, `minimum` of them at least, and each barred gate's opening and its rotating bar's half
// width, by the file's gates.
struct TurnGeometry
{
  struct Gate
  {
    bool present = false;
    double opening_half_m = 0.0;
    double half_width_m = 0.0;
  };
  double window_s = 0.0;
  int minimum = 0;
  std::vector<Gate> gates;
};

// What a takeover file says defines its networks.
struct TakeoverSpec
{
  std::string file;
  double dt = 0.0;
  std::string main_checkpoint;
  std::string main_sha256;
  double aim_m = 0.0;
  double authority = 0.0;
  double ramp_in_s = 0.0;
  double ramp_out_s = 0.0;
  double handback_m = 0.0;
  // Not negative: the hand-back this far past the plane of the gate after the obstacle's, once that
  // gate is passed, in place of handback_m. Negative: handback_m.
  double handback_next_m = -1.0;
  // Positive: at a rotating bar the aim's lane is kept from this long before its plane (LaneHold).
  // Zero: the aim as it says it.
  double lane_hold_s = 0.0;
  // Positive: at a sliding bar the side of the bar the aim names is kept from this long before its
  // plane, at least as deep as that lane's middle (SlideCommit). Zero: the aim as it says it.
  double slide_commit_s = 0.0;
  // The commit's lane runs from the bar's edge to the opening less this.
  double slide_inset_m = 0.0;
  // With slide_commit_s, where the commit's lanes are.
  std::optional<SlideGeometry> slide;
  // Set: at a rotating bar, once the lane hold holds its lane, the aim at least as deep as the middle of the
  // half-aperture on its side of the bar's line at the crossing, from the bar's edge to the opening less
  // this (turnFloor). Unset: the held aim as it is.
  std::optional<double> turn_floor_inset_m;
  // With turn_floor_inset_m, where the floor's lanes are.
  std::optional<TurnGeometry> turn;
  int history = 0;
  double reading_hz = 0.0;
  // Positive: the networks trained on the detector's every frame in the newest slot, the history
  // moving at reading_hz. Zero: one frame per reading.
  double frame_hz = 0.0;
  double max_age_s = 0.0;
  double time_scale_s = 0.0;
  // The residual also reads g * aim, in two columns before the gain.
  bool show_aim = false;
  std::vector<TakeoverGate> gates;
  PolicySource source;
  std::string aim_checkpoint;
  std::string aim_sha256;

  // The row's entries: what the aim reads.
  int width() const {return kMainRowSize + kActionSize + readingEntries() + kBarKinds + 2;}

  // What the residual reads.
  int residualWidth() const {return width() + (show_aim ? kAimSize : 0);}

  int readingEntries() const {return kReadingColumns * history;}
};

// quadlab.envs.takeover.MemoryAim: a tanh layer on standardized columns of the row, a GRU as torch
// computes it, a linear head. The memory is the caller's.
struct MemoryAim
{
  std::vector<Eigen::Index> columns;
  Eigen::VectorXd mean;
  Eigen::VectorXd std;
  MlpPolicy::Layer input;
  // The GRU's weights on its input and on its memory: reset, update and new, in torch's order.
  MlpPolicy::Layer from_input;
  MlpPolicy::Layer from_memory;
  MlpPolicy::Layer head;

  Eigen::Index size() const {return input.weight.rows();}

  // One tick: moves `memory` on and returns the aim, clipped to [-1, 1].
  Eigen::Vector2d step(const Eigen::VectorXd & row, Eigen::VectorXd & memory) const;
};

class TakeoverNetworks
{
public:
  TakeoverNetworks(
    TakeoverSpec spec, std::vector<MlpPolicy::Layer> aim, std::vector<MlpPolicy::Layer> residual);

  TakeoverNetworks(TakeoverSpec spec, MemoryAim aim, std::vector<MlpPolicy::Layer> residual);

  // Parses a quadlab_takeover file and refuses it unless its goldens reproduce.
  static std::shared_ptr<const TakeoverNetworks> load(const std::string & path);

  const TakeoverSpec & spec() const {return spec_;}

  // The aim keeps a memory.
  bool recurrent() const {return memory_aim_.has_value();}

  // Its memory's size; 0 for an aim without one.
  Eigen::Index memorySize() const {return recurrent() ? memory_aim_->size() : 0;}

  // Both clipped to [-1, 1]. A recurrent aim moves `memory` on a tick; the other ignores it.
  Eigen::Vector2d aim(const Eigen::VectorXd & row, Eigen::VectorXd & memory) const;

  // The residual's own row, residualWidth() wide.
  Action residual(const Eigen::VectorXd & row) const;

  const MlpPolicy::GoldenCheck & goldenCheck() const {return golden_check_;}

private:
  TakeoverSpec spec_;
  std::vector<MlpPolicy::Layer> aim_;
  std::optional<MemoryAim> memory_aim_;
  std::vector<MlpPolicy::Layer> residual_;
  MlpPolicy::GoldenCheck golden_check_;
};

// The bar kinds of the file's gates, in its order, from the gates a flight names and what stands
// in each. Throws std::invalid_argument unless they are the file's gates, each with a bar it
// trained on or none: a gate left out is a mistake, a gate without its bar is said.
std::vector<BarKind> barKinds(
  const TakeoverSpec & spec, const std::vector<std::string> & gates,
  const std::vector<std::string> & kinds);

// The last readings of one obstacle, newest first. A detector faster than the rate the networks
// trained on is thinned to it: at its own rate, the history would span a fraction of the time
// they read the bar's motion over. With `latest`, a frame between two steps of the history
// replaces the newest reading instead of being dropped, so the newest is always the latest frame.
class BarReadings
{
public:
  BarReadings(int history, double hz, double max_age_s, bool latest = false);

  void clear();

  // A reading stamped with its frame's time; false when late, repeated or thinned out, which a
  // reading with `latest` never is.
  bool add(double stamp, const std::array<double, 2> & values);

  // A reading kept whatever its time, past the thinning.
  void keep(double stamp, const std::array<double, 2> & values);

  // Newest first, (value, value, age over max_age_s): zeros and 1 for a reading gone or missing.
  void write(double now, Eigen::Ref<Eigen::VectorXd> out) const;

  // The time of the newest reading; -infinity with none.
  double newest() const {return stamps_.front();}

private:
  double period_s_;
  double max_age_s_;
  bool latest_;
  double next_due_ = -std::numeric_limits<double>::infinity();
  std::vector<double> stamps_;
  std::vector<std::array<double, 2>> values_;
};

// An aim this long or more names a lane, in its units: quadlab.envs.takeover.HOLD_MIN.
inline constexpr double kHoldMin = 0.2;

// quadlab.envs.takeover.TakeoverVecEnv._hold_lane: at a rotating bar, in the window and before its
// plane, from hold_s before the plane, the aim's lane is kept. Its half-plane is the one of the last
// held aim that named a lane: an aim across it flies reversed, the lane on the other side of the
// bar's centre, so a late flip of the aim moves nothing; a lane turning with the bar turns the hold.
class LaneHold
{
public:
  // One tick's aim as flown. `on`: a rotating bar, the window active, before its plane; the hold
  // ends on the first tick it is not.
  Eigen::Vector2d hold(const Eigen::Vector2d & aim, bool on, double tau_s, double hold_s);

  void clear() {on_ = false;}

  bool on() const {return on_;}

private:
  bool on_ = false;
  Eigen::Vector2d ref_ = Eigen::Vector2d::Zero();
};

// quadlab.envs.takeover.TakeoverVecEnv._commit_slide: at a sliding bar, in the window and before its
// plane, from slide_commit_s before the plane, the side of the bar the aim names then is kept, at least
// as deep as the middle of that lane: from the bar's edge to the opening less slide_inset_m, the bar
// where a line through the row's readings puts it at the crossing, deeper if the aim says so. A
// near-central bar leaves two lanes alike, and the aim hedges between them: shallow, and late back
// over the bar.
class SlideCommit
{
public:
  // One tick's aim as flown, `row` the aim's as trained. `on`: a sliding bar of `kind` at the file's
  // `gate`, the window active, before its plane; the commit ends on the first tick it is not.
  Eigen::Vector2d commit(
    const Eigen::Vector2d & aim, bool on, BarKind kind, std::size_t gate, const Eigen::VectorXd & row,
    const TakeoverSpec & spec);

  void clear() {on_ = false;}

  bool on() const {return on_;}

  // The kept side, +1 or -1 along the bar's run (a left-right bar's along the gate's right).
  double side() const {return side_;}

private:
  bool on_ = false;
  double side_ = 0.0;
};

// quadlab.envs.takeover.TakeoverVecEnv._floor_turn: at a rotating bar, once the lane hold holds its lane,
// the aim at least as deep as the middle of the half-aperture on its side of the bar's line, from the bar's
// edge to the opening less turn_floor_inset_m, the line where one through the row's angle readings puts the
// bar at the crossing; the aim's part along the bar as it says. As at a sliding bar, a lane near the bar
// hedges between two alike lanes. `on`: the hold on, the window active, before the plane, at the file's
// `gate`. Stateless: the hold keeps the side.
Eigen::Vector2d turnFloor(
  const Eigen::Vector2d & aim, bool on, std::size_t gate, const Eigen::VectorXd & row,
  const TakeoverSpec & spec);

// What one tick did, for the debug topic.
struct TakeoverTick
{
  TakeoverMode mode = TakeoverMode::Before;
  std::size_t obstacle = 0;
  double gain = 0.0;
  double tau_s = 0.0;
  Action main{};
  // The aim before the plane, as it says it; zero past it, where nothing is applied.
  Eigen::Vector2d aim = Eigen::Vector2d::Zero();
  Action aimed{};
  Action residual{};
  // The row the residual read; empty outside the window.
  Eigen::VectorXd row;
};

class Takeover
{
public:
  // kinds: what stands at each of the file's gates, barKinds().
  Takeover(
    std::shared_ptr<const TakeoverNetworks> networks, std::vector<BarKind> kinds, BarSide one_at);

  // The race starts on a course of so many gates: the next obstacle from the target, nothing read
  // of it.
  void start(std::size_t target, std::size_t gates);

  // One detector result, kept when it is of the next obstacle. Any time, the frame's stamp.
  bool addReading(const std::string & gate, double position, double stamp);

  // A reading in the row's values, past the thinning and the gate.
  void keepValues(double stamp, const std::array<double, 2> & values);

  // A reading in the row's values, past the gate alone.
  bool addValues(double stamp, const std::array<double, 2> & values);

  // This tick's window, on the estimate and the target.
  void advance(const VehicleState & state, const Course & course, std::size_t target);

  // This tick's action, after advance(). Outside the window, exactly one act() of main.
  Action act(
    const Policy & main, const Eigen::VectorXd & row, const VehicleState & state,
    const Course & course, double t);

  const TakeoverNetworks & networks() const {return *networks_;}

  const std::vector<BarKind> & kinds() const {return kinds_;}

  BarSide oneAt() const {return one_at_;}

  TakeoverMode mode() const {return mode_;}

  // The next obstacle, an index of the file's gates.
  std::size_t obstacle() const {return obstacle_;}

  double gain() const {return gain_;}

  const TakeoverTick & tick() const {return tick_;}

  const BarReadings & readings() const {return readings_;}

  // A recurrent aim's memory: zeros away from the obstacle.
  const Eigen::VectorXd & memory() const {return memory_;}

  // From the target being the gate before the obstacle to the hand-back's end: the approach a
  // recurrent aim remembers.
  bool near() const;

  // With the file's lane_hold_s, the rotating bar's lane is held.
  const LaneHold & laneHold() const {return hold_;}

  // With the file's slide_commit_s, a sliding bar's lane is committed.
  const SlideCommit & slideCommit() const {return slide_;}

  // Detector results taken in, of any gate, and how many of them the row has held.
  std::uint64_t readingsSeen() const {return seen_;}

  std::uint64_t readingsKept() const {return kept_;}

private:
  // The first barred gate at or after the target, round the course.
  std::size_t nextObstacle(std::size_t target, std::size_t gates) const;

  void open();

  // How far past the obstacle's plane, along its normal.
  double beyond(const VehicleState & state, const Course & course) const;

  // How far past the plane of the gate after the obstacle's, along that gate's normal.
  double beyondNext(const VehicleState & state, const Course & course) const;

  // To the obstacle's plane at the speed along its normal, 1 m/s at least, in [0, time_scale_s].
  double tau(const VehicleState & state, const Course & course) const;

  std::shared_ptr<const TakeoverNetworks> networks_;
  std::vector<BarKind> kinds_;
  BarSide one_at_;
  BarReadings readings_;
  TakeoverMode mode_ = TakeoverMode::Before;
  std::size_t obstacle_ = 0;
  // The last advance()'s.
  std::size_t target_ = 0;
  std::size_t gates_ = 0;
  double clock_s_ = 0.0;
  double gain_ = 0.0;
  double gain_out_ = 0.0;
  Eigen::VectorXd memory_;
  LaneHold hold_;
  SlideCommit slide_;
  TakeoverTick tick_;
  std::uint64_t seen_ = 0;
  std::uint64_t kept_ = 0;
};

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__TAKEOVER_HPP_
