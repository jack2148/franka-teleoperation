# Public v1 source/reference contract

`franka-teleoperation` publishes control source and software-validation evidence. It is not an executable reproduction release of the historical OMY system.

## Current / validated software scope

Ordinary Geomagic Touch device/bridge source, Cartesian position/orientation mapping, clutch/anchor behavior, DLS/null-space related code, target/command conditioning, public haptic implementation, CPU regression tests, control-equivalence evidence, and architecture/provenance documentation are included. Physical device acceptance and physical Franka deployment are not established by those tests.

## Historical source scope

OMY-L100 source, Cartesian retargeting, DLS inverse kinematics, null-space control, clutch/command conditioning, diagnostics/collision-feedback source, and core mathematical tests remain. Original launcher/config paths are historical reference material. The scene/model bundle was not committed, exact native provenance was not recovered, and modified derivative copies are not substituted. No ready-to-run original OMY environment is supplied.

## Companion boundary

The separately approved `franka-imitation-learning` candidate retains Square, ordinary Touch collection, recording, dataset validation, LeRobot conversion, ACT/Diffusion Policy wrappers, fixed TEST50 result/protocol, spatial/failure analysis and historical TRAIN limitations. It is unchanged. Its executable/research-artifact scope depends on documented external data, models, packages and hardware where relevant. Old OMY executable-gate notes describe the historical interface; they do not make the missing OMY scene a dependency of ordinary Touch Square collection.

## Exclusions and invariants

Unreleased alternative controllers, device/task extensions and weighting pipelines remain outside v1. Their absence is a release boundary, not a skipped or mocked test result. No further ownership investigation of excluded code is performed.

Existing synthetic tests and the explicit SDK-dependent hardware-validation boundary remain intact. Source cutoff, gains, coordinate mappings, DLS/null-space implementation, conditioning and clutch are unchanged. No common controller is extracted. Later research-workspace changes do not auto-sync into this candidate. Learning algorithms, dataset contracts, success criteria, training configurations and TEST50 results are untouched.
