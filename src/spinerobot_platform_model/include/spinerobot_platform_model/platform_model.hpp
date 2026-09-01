#pragma once

#include <array>

namespace spinerobot::platform
{

struct PlatformPose4D
{
  double x_m{0.0};
  double y_m{0.0};
  double pitch_rad{0.0};
  double yaw_rad{0.0};
};

struct ActuatorState
{
  std::array<double, 4> position_m{};
};

class PlatformModel
{
public:
  ActuatorState inverseKinematics(const PlatformPose4D& pose) const;

  PlatformPose4D forwardKinematics(
      const ActuatorState& actuators,
      const PlatformPose4D& initial_guess) const;

  bool isInsideWorkspace(const PlatformPose4D& pose) const;
  bool actuatorLimitsValid(const ActuatorState& actuators) const;
  double singularityMetric(const PlatformPose4D& pose) const;
  double constraintResidual(
      const PlatformPose4D& pose,
      const ActuatorState& actuators) const;
};

}  // namespace spinerobot::platform
