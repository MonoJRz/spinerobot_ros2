# Logical Architecture

- UI / medical visualization
- Procedure HFSM
- Core surgical services
- ROS 2 / DDS middleware
- Hardware abstraction
- Physical hardware

Safety is cross-cutting:
- procedure guards
- tracking/registration validity
- motion command gate
- driver/controller limits
- physical E-stop chain

## Closed-chain platform rule

Do not force the physical parallel mechanism into the tf2 tree.

Publish only the functional transform:

`platform_base -> moving_platform -> guide`

The closed-chain constraints are solved by `spinerobot_platform_model`.
