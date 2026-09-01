#include "spinerobot_platform_model/platform_model.hpp"

#include <stdexcept>

namespace spinerobot::platform
{

ActuatorState PlatformModel::inverseKinematics(const PlatformPose4D&) const
{
  throw std::logic_error("TODO: implement closed-chain inverse kinematics from CAD geometry.");
}

PlatformPose4D PlatformModel::forwardKinematics(
    const ActuatorState&,
    const PlatformPose4D&) const
{
  throw std::logic_error("TODO: implement numerical closed-chain forward kinematics.");
}

bool PlatformModel::isInsideWorkspace(const PlatformPose4D&) const
{
  return false;
}

bool PlatformModel::actuatorLimitsValid(const ActuatorState&) const
{
  return false;
}

double PlatformModel::singularityMetric(const PlatformPose4D&) const
{
  return 0.0;
}

double PlatformModel::constraintResidual(
    const PlatformPose4D&,
    const ActuatorState&) const
{
  return 0.0;
}

}  // namespace spinerobot::platform
