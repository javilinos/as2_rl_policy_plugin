#include "as2_rl_policy/core/takeover.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "as2_rl_policy/core/decode.hpp"
#include "yaml_fields.hpp"

namespace rl_policy
{

namespace
{

// A frame this early on the thinning's grid is on it: a stream at the trained rate is kept whole.
constexpr double kThinningToleranceS = 1.0e-6;
// A reading at its maximum age is gone, however its age rounds, quadlab's AGE_ROUNDING_S.
constexpr double kAgeRoundingS = 1.0e-9;

std::vector<MlpPolicy::Layer> parseNetwork(
  const YAML::Node & root, const std::string & name, Eigen::Index inputs, Eigen::Index outputs)
{
  const YAML::Node nodes = required(required(root, name, ""), "layers", name);
  if (!nodes.IsSequence() || nodes.size() == 0) {
    throw PolicyError("'" + name + ".layers' must be a non-empty sequence");
  }
  std::vector<MlpPolicy::Layer> layers;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const std::string path = name + ".layers[" + std::to_string(i) + "]";
    layers.push_back(parseLayer(nodes[i], path));
    if (layers.back().weight.cols() != inputs) {
      throw PolicyError(
              "'" + path + "' takes " + std::to_string(layers.back().weight.cols()) +
              " inputs, the previous stage gives " + std::to_string(inputs));
    }
    if (!layers.back().weight.allFinite() || !layers.back().bias.allFinite()) {
      throw PolicyError("'" + path + "' has non-finite parameters");
    }
    inputs = layers.back().weight.rows();
  }
  if (inputs != outputs) {
    throw PolicyError(
            "'" + name + "' gives " + std::to_string(inputs) + " outputs, not " +
            std::to_string(outputs));
  }
  return layers;
}

TakeoverGate parseGate(const YAML::Node & node, const std::string & path)
{
  TakeoverGate gate;
  gate.name = scalar<std::string>(required(node, "name", path), path + ".name", "string");
  const int index = scalar<int>(required(node, "index", path), path + ".index", "integer");
  if (index < 0) {
    throw PolicyError("'" + path + ".index' must not be negative");
  }
  gate.index = static_cast<std::size_t>(index);
  gate.open_s = finiteDouble(required(node, "open_s", path), path + ".open_s");
  if (gate.open_s < 0.0) {
    throw PolicyError("'" + path + ".open_s' must not be negative");
  }
  const YAML::Node kinds = required(node, "kinds", path);
  if (!kinds.IsSequence() || kinds.size() == 0) {
    throw PolicyError("'" + path + ".kinds' must be a non-empty sequence");
  }
  for (std::size_t i = 0; i < kinds.size(); ++i) {
    const std::string kind_path = path + ".kinds[" + std::to_string(i) + "]";
    try {
      gate.kinds.push_back(
        barKindFromString(scalar<std::string>(kinds[i], kind_path, "string")));
    } catch (const std::invalid_argument & e) {
      throw PolicyError("'" + kind_path + "': " + e.what());
    }
  }
  return gate;
}

Action actionFrom(const Eigen::VectorXd & outputs)
{
  Action action{};
  for (int k = 0; k < kActionSize; ++k) {
    action[k] = std::clamp(outputs(k), -1.0, 1.0);
  }
  return action;
}

Eigen::VectorXd asTrained(const Eigen::VectorXd & row)
{
  return row.cast<float>().cast<double>();
}

}  // namespace

const char * toString(BarKind kind)
{
  switch (kind) {
    case BarKind::UpDown:
      return "vertical";
    case BarKind::LeftRight:
      return "horizontal";
    case BarKind::Rotating:
      return "rotating";
    case BarKind::None:
      return "none";
  }
  return "unknown";
}

const char * toString(BarSide side)
{
  return side == BarSide::Left ? "left" : "right";
}

const char * toString(TakeoverMode mode)
{
  switch (mode) {
    case TakeoverMode::Before:
      return "before";
    case TakeoverMode::Active:
      return "active";
    case TakeoverMode::HandingBack:
      return "handing_back";
    case TakeoverMode::Tail:
      return "tail";
  }
  return "unknown";
}

BarKind barKindFromString(const std::string & name)
{
  if (name == "vertical") {
    return BarKind::UpDown;
  }
  if (name == "horizontal") {
    return BarKind::LeftRight;
  }
  if (name == "rotating") {
    return BarKind::Rotating;
  }
  if (name == "none") {
    return BarKind::None;
  }
  throw std::invalid_argument("'" + name + "' is none of vertical, horizontal, rotating, none");
}

BarSide barSideFromString(const std::string & name)
{
  if (name == "left") {
    return BarSide::Left;
  }
  if (name == "right") {
    return BarSide::Right;
  }
  throw std::invalid_argument("'" + name + "' is neither left nor right");
}

std::array<double, 2> barValues(BarKind kind, double position, BarSide one_at)
{
  if (kind == BarKind::Rotating) {
    const double angle = position * kBarTurnRad;
    return {std::sin(2.0 * angle), std::cos(2.0 * angle)};
  }
  const double place = 2.0 * position - 1.0;
  return {kind == BarKind::LeftRight && one_at == BarSide::Left ? -place : place, 0.0};
}

TakeoverNetworks::TakeoverNetworks(
  TakeoverSpec spec, std::vector<MlpPolicy::Layer> aim, std::vector<MlpPolicy::Layer> residual)
: spec_(std::move(spec)), aim_(std::move(aim)), residual_(std::move(residual))
{
}

std::shared_ptr<const TakeoverNetworks> TakeoverNetworks::load(const std::string & path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::BadFile &) {
    throw PolicyError("cannot open takeover file '" + path + "'");
  } catch (const YAML::Exception & e) {
    throw PolicyError("takeover file '" + path + "' is not valid YAML: " + e.what());
  }

  try {
    expectString(root, "format", "", kTakeoverFormat);
    const int version =
      scalar<int>(required(root, "format_version", ""), "format_version", "integer");
    if (version < 1 || version > kTakeoverFormatVersion) {
      throw PolicyError(
              "format_version " + std::to_string(version) + " is not supported, only up to " +
              std::to_string(kTakeoverFormatVersion));
    }

    TakeoverSpec spec;
    spec.file = path;
    spec.dt = positiveDouble(required(root, "dt", ""), "dt");
    const YAML::Node main = required(root, "main", "");
    spec.main_checkpoint = optionalString(main, "checkpoint", "unknown");
    spec.main_sha256 = scalar<std::string>(
      required(main, "checkpoint_sha256", "main"), "main.checkpoint_sha256", "string");

    const YAML::Node window = required(root, "window", "");
    const auto positive = [&window](const char * key) {
        return positiveDouble(required(window, key, "window"), std::string("window.") + key);
      };
    spec.aim_m = positive("aim_m");
    spec.authority = positive("authority");
    spec.ramp_in_s = positive("ramp_in_s");
    spec.ramp_out_s = positive("ramp_out_s");
    spec.handback_m = positive("handback_m");

    const YAML::Node reading = required(root, "reading", "");
    spec.history =
      scalar<int>(required(reading, "history", "reading"), "reading.history", "integer");
    if (spec.history < 1) {
      throw PolicyError("'reading.history' must be at least 1");
    }
    spec.reading_hz = positiveDouble(required(reading, "hz", "reading"), "reading.hz");
    const YAML::Node frames = reading["frame_hz"];
    if (frames.IsDefined() && !frames.IsNull()) {
      spec.frame_hz = finiteDouble(frames, "reading.frame_hz");
      if (spec.frame_hz < 0.0) {
        throw PolicyError("'reading.frame_hz' must not be negative");
      }
    }
    spec.max_age_s = positiveDouble(required(reading, "max_age_s", "reading"), "reading.max_age_s");
    spec.time_scale_s =
      positiveDouble(required(reading, "time_scale_s", "reading"), "reading.time_scale_s");

    const YAML::Node gates = required(root, "gates", "");
    if (!gates.IsSequence() || gates.size() < 2) {
      throw PolicyError("'gates' must hold two gates at least: the window moves between them");
    }
    for (std::size_t i = 0; i < gates.size(); ++i) {
      spec.gates.push_back(parseGate(gates[i], "gates[" + std::to_string(i) + "]"));
    }

    const YAML::Node source = root["source"];
    spec.source.checkpoint = optionalString(source, "checkpoint", "unknown");
    spec.source.checkpoint_sha256 = optionalString(source, "checkpoint_sha256", "unknown");
    spec.source.env_config = optionalString(source, "env_config", "unknown");
    spec.source.plant_config = optionalString(source, "plant_config", "unknown");
    spec.source.quadlab_commit = optionalString(source, "quadlab_commit", "unknown");
    spec.aim_checkpoint = optionalString(source, "aim_checkpoint", "unknown");
    spec.aim_sha256 = optionalString(source, "aim_checkpoint_sha256", "unknown");

    const int width = spec.width();
    auto networks = std::make_shared<TakeoverNetworks>(
      std::move(spec), parseNetwork(root, "aim", width, kAimSize),
      parseNetwork(root, "residual", width, kActionSize));

    const YAML::Node golden = required(root, "golden", "");
    const double tolerance =
      finiteDouble(required(golden, "tolerance", "golden"), "golden.tolerance");
    if (tolerance < 0.0) {
      throw PolicyError("'golden.tolerance' must not be negative");
    }
    const auto inputs =
      rows(required(golden, "input", "golden"), "golden.input", static_cast<std::size_t>(width));
    const auto aims = rows(required(golden, "aim", "golden"), "golden.aim", kAimSize);
    const auto residuals =
      rows(required(golden, "residual", "golden"), "golden.residual", kActionSize);
    if (inputs.size() != aims.size() || inputs.size() != residuals.size()) {
      throw PolicyError(
              "'golden.input' has " + std::to_string(inputs.size()) + " rows, 'golden.aim' " +
              std::to_string(aims.size()) + ", 'golden.residual' " +
              std::to_string(residuals.size()));
    }
    if (inputs.empty()) {
      throw PolicyError("'golden' carries no rows, the networks cannot be checked");
    }

    MlpPolicy::GoldenCheck check;
    check.rows = inputs.size();
    check.tolerance = tolerance;
    for (std::size_t row = 0; row < inputs.size(); ++row) {
      const Eigen::VectorXd input = Eigen::Map<const Eigen::VectorXd>(
        inputs[row].data(), static_cast<Eigen::Index>(inputs[row].size()));
      const Eigen::Vector2d aim = networks->aim(input);
      const Action residual = networks->residual(input);
      double error = std::max(std::abs(aim.x() - aims[row][0]), std::abs(aim.y() - aims[row][1]));
      for (int k = 0; k < kActionSize; ++k) {
        error = std::max(error, std::abs(residual[k] - residuals[row][k]));
      }
      check.max_error = std::max(check.max_error, error);
      if (error > tolerance) {
        throw PolicyError(
                "golden row " + std::to_string(row) + " misses by " + number(error) +
                ", more than the tolerance " + number(tolerance) +
                ": the weights do not reproduce the exported networks");
      }
    }
    networks->golden_check_ = check;
    return networks;
  } catch (const PolicyError & e) {
    throw PolicyError("takeover file '" + path + "': " + e.what());
  }
}

Eigen::Vector2d TakeoverNetworks::aim(const Eigen::VectorXd & row) const
{
  return forward(aim_, row).cwiseMax(-1.0).cwiseMin(1.0);
}

Action TakeoverNetworks::residual(const Eigen::VectorXd & row) const
{
  return actionFrom(forward(residual_, row));
}

std::vector<BarKind> barKinds(
  const TakeoverSpec & spec, const std::vector<std::string> & gates,
  const std::vector<std::string> & kinds)
{
  if (gates.size() != kinds.size()) {
    throw std::invalid_argument(
            "the flight names " + std::to_string(gates.size()) + " gates and " +
            std::to_string(kinds.size()) + " bar kinds");
  }
  std::vector<BarKind> out;
  for (const TakeoverGate & gate : spec.gates) {
    const auto it = std::find(gates.begin(), gates.end(), gate.name);
    if (it == gates.end()) {
      throw std::invalid_argument(
              "the networks trained around a bar in '" + gate.name + "', which the flight does "
              "not name");
    }
    const std::size_t at = static_cast<std::size_t>(it - gates.begin());
    const BarKind kind = barKindFromString(kinds[at]);
    if (kind != BarKind::None &&
      std::find(gate.kinds.begin(), gate.kinds.end(), kind) == gate.kinds.end())
    {
      throw std::invalid_argument(
              "the networks did not train on a " + kinds[at] + " bar in '" + gate.name + "'");
    }
    out.push_back(kind);
  }
  if (gates.size() != spec.gates.size()) {
    throw std::invalid_argument(
            "the flight names " + std::to_string(gates.size()) + " gates, the networks trained "
            "around " + std::to_string(spec.gates.size()));
  }
  return out;
}

BarReadings::BarReadings(int history, double hz, double max_age_s, bool latest)
: period_s_(1.0 / hz), max_age_s_(max_age_s), latest_(latest)
{
  stamps_.resize(static_cast<std::size_t>(history));
  values_.resize(static_cast<std::size_t>(history));
  clear();
}

void BarReadings::clear()
{
  next_due_ = -std::numeric_limits<double>::infinity();
  std::fill(stamps_.begin(), stamps_.end(), -std::numeric_limits<double>::infinity());
  std::fill(values_.begin(), values_.end(), std::array<double, 2>{});
}

bool BarReadings::add(double stamp, const std::array<double, 2> & values)
{
  if (!(stamp > stamps_.front())) {
    return false;
  }
  if (stamp < next_due_ - kThinningToleranceS) {
    if (latest_) {
      stamps_.front() = stamp;
      values_.front() = values;
    }
    return latest_;
  }
  next_due_ += period_s_;
  if (next_due_ <= stamp) {
    next_due_ = stamp + period_s_;
  }
  keep(stamp, values);
  return true;
}

void BarReadings::keep(double stamp, const std::array<double, 2> & values)
{
  std::rotate(stamps_.rbegin(), stamps_.rbegin() + 1, stamps_.rend());
  std::rotate(values_.rbegin(), values_.rbegin() + 1, values_.rend());
  stamps_.front() = stamp;
  values_.front() = values;
}

void BarReadings::write(double now, Eigen::Ref<Eigen::VectorXd> out) const
{
  for (std::size_t i = 0; i < stamps_.size(); ++i) {
    const double age = std::max(now - stamps_[i], 0.0);
    const bool fresh = age < max_age_s_ - kAgeRoundingS;
    const Eigen::Index at = static_cast<Eigen::Index>(i) * kReadingColumns;
    out(at) = fresh ? values_[i][0] : 0.0;
    out(at + 1) = fresh ? values_[i][1] : 0.0;
    out(at + 2) = std::min(age, max_age_s_) / max_age_s_;
  }
}

Takeover::Takeover(
  std::shared_ptr<const TakeoverNetworks> networks, std::vector<BarKind> kinds, BarSide one_at)
: networks_(std::move(networks)), kinds_(std::move(kinds)), one_at_(one_at),
  readings_(
    networks_->spec().history, networks_->spec().reading_hz, networks_->spec().max_age_s,
    networks_->spec().frame_hz > 0.0)
{
  if (kinds_.size() != networks_->spec().gates.size()) {
    throw std::invalid_argument(
            "the takeover has " + std::to_string(kinds_.size()) + " bar kinds for " +
            std::to_string(networks_->spec().gates.size()) + " gates");
  }
  if (std::all_of(
      kinds_.begin(), kinds_.end(), [](BarKind kind) {return kind == BarKind::None;}))
  {
    throw std::invalid_argument("the takeover has no bar in any of its gates");
  }
}

void Takeover::start(std::size_t target, std::size_t gates)
{
  obstacle_ = nextObstacle(target, gates);
  open();
}

std::size_t Takeover::nextObstacle(std::size_t target, std::size_t gates) const
{
  std::size_t next = 0;
  std::size_t best = gates;
  const auto & barred = networks_->spec().gates;
  for (std::size_t i = 0; i < barred.size(); ++i) {
    const std::size_t ahead = (barred[i].index + gates - target) % gates;
    if (kinds_[i] != BarKind::None && ahead < best) {
      best = ahead;
      next = i;
    }
  }
  return next;
}

void Takeover::open()
{
  mode_ = TakeoverMode::Before;
  gain_ = 0.0;
  clock_s_ = 0.0;
  readings_.clear();
}

bool Takeover::addReading(const std::string & gate, double position, double stamp)
{
  ++seen_;
  if (gate != networks_->spec().gates[obstacle_].name || !std::isfinite(position) ||
    position < 0.0 || position > 1.0 || !std::isfinite(stamp))
  {
    return false;
  }
  if (!readings_.add(stamp, barValues(kinds_[obstacle_], position, one_at_))) {
    return false;
  }
  ++kept_;
  return true;
}

void Takeover::keepValues(double stamp, const std::array<double, 2> & values)
{
  readings_.keep(stamp, values);
}

bool Takeover::addValues(double stamp, const std::array<double, 2> & values)
{
  return readings_.add(stamp, values);
}

double Takeover::beyond(const VehicleState & state, const Course & course) const
{
  const Gate & gate = course.gate(networks_->spec().gates[obstacle_].index);
  const Eigen::Vector3d normal(std::cos(gate.yaw), std::sin(gate.yaw), 0.0);
  return (state.position - gate.position).dot(normal);
}

double Takeover::tau(const VehicleState & state, const Course & course) const
{
  const Gate & gate = course.gate(networks_->spec().gates[obstacle_].index);
  const Eigen::Vector3d normal(std::cos(gate.yaw), std::sin(gate.yaw), 0.0);
  const double speed = std::max(state.velocity.dot(normal), 1.0);
  return std::clamp(-beyond(state, course) / speed, 0.0, networks_->spec().time_scale_s);
}

void Takeover::advance(const VehicleState & state, const Course & course, std::size_t target)
{
  const TakeoverSpec & spec = networks_->spec();
  const std::size_t gates = course.size();
  const TakeoverGate & gate = spec.gates[obstacle_];
  clock_s_ += spec.dt;

  const bool early = target == (gate.index + gates - 1) % gates && tau(state, course) < gate.open_s;
  if (mode_ == TakeoverMode::Before && (target == gate.index || early)) {
    mode_ = TakeoverMode::Active;
    clock_s_ = 0.0;
  }
  const std::size_t past = (target + gates - gate.index) % gates;
  if (mode_ == TakeoverMode::Active && past >= 1 && past < gates / 2 &&
    beyond(state, course) > spec.handback_m)
  {
    mode_ = TakeoverMode::HandingBack;
    clock_s_ = 0.0;
    gain_out_ = gain_;
  }
  if (mode_ == TakeoverMode::Active) {
    gain_ = std::min(clock_s_ / spec.ramp_in_s, 1.0);
  } else if (mode_ == TakeoverMode::HandingBack) {
    gain_ = gain_out_ * std::max(1.0 - clock_s_ / spec.ramp_out_s, 0.0);
    if (gain_ <= 0.0) {
      mode_ = TakeoverMode::Tail;
    }
  }
  if (mode_ == TakeoverMode::Tail) {
    // With one bar standing the next obstacle is the same gate, a lap on.
    obstacle_ = nextObstacle(target, gates);
    open();
  }
}

Action Takeover::act(
  const Policy & main, const Eigen::VectorXd & row, const VehicleState & state,
  const Course & course, double t)
{
  const TakeoverSpec & spec = networks_->spec();
  tick_ = TakeoverTick();
  tick_.mode = mode_;
  tick_.obstacle = obstacle_;
  tick_.gain = gain_;
  tick_.tau_s = tau(state, course);
  tick_.main = clip(main.act(row));
  tick_.aimed = tick_.main;
  if (!(gain_ > 0.0)) {
    return tick_.main;
  }

  const int width = spec.width();
  Eigen::VectorXd input = Eigen::VectorXd::Zero(width);
  input.head(kMainRowSize) = row;
  for (int k = 0; k < kActionSize; ++k) {
    input(kMainRowSize + k) = tick_.main[k];
  }
  const Eigen::Index readings = kMainRowSize + kActionSize;
  readings_.write(t, input.segment(readings, spec.readingEntries()));
  input(readings + spec.readingEntries() + static_cast<int>(kinds_[obstacle_])) = 1.0;
  input(width - 2) = tick_.tau_s / spec.time_scale_s;
  input(width - 1) = gain_;
  input = asTrained(input);

  // The aim's job ends at the plane, and past it it was never taught.
  if (beyond(state, course) < 0.0) {
    tick_.aim = networks_->aim(input);
    Eigen::VectorXd shifted = row;
    shifted.segment<kAimSize>(1) -= spec.aim_m * gain_ * tick_.aim;
    tick_.aimed = clip(main.act(asTrained(shifted)));
    for (int k = 0; k < kActionSize; ++k) {
      input(kMainRowSize + k) = static_cast<float>(tick_.aimed[k]);
    }
  }
  tick_.residual = networks_->residual(input);
  tick_.row = input;

  Action action{};
  for (int k = 0; k < kActionSize; ++k) {
    action[k] = std::clamp(tick_.aimed[k] + spec.authority * gain_ * tick_.residual[k], -1.0, 1.0);
  }
  return action;
}

}  // namespace rl_policy
