#ifndef AS2_RL_POLICY__CORE__POLICY_HPP_
#define AS2_RL_POLICY__CORE__POLICY_HPP_

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl_policy
{

using Action = std::array<double, 4>;

inline constexpr int kActionSize = 4;
inline constexpr int kGateRelativeSize = 20;
inline constexpr const char kPolicyFormat[] = "quadlab_policy";
inline constexpr int kPolicyFormatVersion = 1;
inline constexpr const char kGateRelativeLayout[] = "gate_relative_20";
inline constexpr const char kBodyRatesContract[] = "indi_body_rates_v1";
inline constexpr const char kTanhActivation[] = "tanh";
inline constexpr const char kClipOutput[] = "clip";

struct ActionSpec
{
  std::string contract;
  double specific_thrust_max_ms2 = 0.0;
  std::array<double, 3> max_body_rate_rad_s{};
};

struct ObservationSpec
{
  std::string layout;
  int action_history = 0;
  int dim = 0;
  double motor_speed_normalization = 0.0;
};

struct PolicySource
{
  std::string checkpoint;
  std::string checkpoint_sha256;
  std::string env_config;
  std::string plant_config;
  std::string quadlab_commit;
};

struct PolicySpec
{
  std::string file;
  std::string task;
  double dt = 0.0;
  ActionSpec action;
  ObservationSpec observation;
  PolicySource source;
};

class PolicyError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

class Policy
{
public:
  virtual ~Policy() = default;

  virtual Action act(const Eigen::VectorXd & observation) const = 0;

  virtual const PolicySpec & spec() const = 0;
};

// Throws PolicyError unless the spec is the gate-relative layout with the body-rate decode.
void checkSupported(const PolicySpec & spec);

class MlpPolicy : public Policy
{
public:
  struct Layer
  {
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
  };

  struct GoldenCheck
  {
    std::size_t rows = 0;
    double max_error = 0.0;
    double tolerance = 0.0;
  };

  MlpPolicy(PolicySpec spec, std::vector<Layer> layers);

  // Parses a quadlab_policy file and refuses it unless its goldens reproduce.
  static std::shared_ptr<MlpPolicy> load(const std::string & path);

  Action act(const Eigen::VectorXd & observation) const override;

  const PolicySpec & spec() const override {return spec_;}

  const std::vector<Layer> & layers() const {return layers_;}

  const GoldenCheck & goldenCheck() const {return golden_check_;}

private:
  PolicySpec spec_;
  std::vector<Layer> layers_;
  GoldenCheck golden_check_;
};

class PolicyBank
{
public:
  static PolicyBank load(
    const std::vector<std::string> & names, const std::vector<std::string> & files);

  void add(const std::string & name, std::shared_ptr<const Policy> policy);

  bool has(const std::string & name) const;

  std::shared_ptr<const Policy> get(const std::string & name) const;

  std::vector<std::string> names() const;

  std::size_t size() const {return policies_.size();}

  double dt() const {return dt_;}

private:
  std::map<std::string, std::shared_ptr<const Policy>> policies_;
  double dt_ = 0.0;
};

}  // namespace rl_policy

#endif  // AS2_RL_POLICY__CORE__POLICY_HPP_
