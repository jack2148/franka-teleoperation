# Validation evidence and limits

Recorded on **2026-10-27 KST** for the source/reference release candidate. A test pass here describes the listed software check; it does not certify hardware or reconstruct missing historical scenes.

## Teleoperation regression rerun

The existing public teleoperation suite was rerun for the release-contract revision:

| Group | Passed | Errors | Failures | Skips |
| --- | ---: | ---: | ---: | ---: |
| Ordinary Geomagic Touch | 672 | 0 | 0 | 0 |
| OMY mathematics | 6 | 0 | 0 | 0 |
| Total | **678** | **0** | **0** | **0** |

Pytest completed in 98.91 seconds. The selected paths were `tests/` and `geomagic/tests/`; no tests were deleted, skipped or replaced for this revision. Four OMY tests execute pure functions from the shipped bridge; two retain the earlier null-space mathematical checks. The test command and paired-checkout dependency requirements are in [INSTALL.md](../INSTALL.md).

The private build record retains the raw JUnit report with SHA256 `e22d6cb8428d76cbf59a6c52f377cd3947cdc5e063233ad60688cd1d48030611`. This public summary avoids embedding host paths from that report. The hash identifies the recorded report, not an additional public download or an installation requirement.

## Preserved control-equivalence evidence

The approved CPU control comparison used four synthetic input modes and compared the frozen source, public candidate and relocated clean tree. Each mode records 700 steps of command, qpos, qvel and returned force values, with 42 values per step.

| Input mode | Recorded shape | Exact equality | Maximum absolute difference |
| --- | --- | --- | ---: |
| Position | 700 × 42 | Yes | 0 |
| Z rotation | 700 × 42 | Yes | 0 |
| Rotation | 700 × 42 | Yes | 0 |
| 6DoF | 700 × 42 | Yes | 0 |

This is **2,800 synthetic CPU steps**, not a physical-device timing or haptic-acceptance experiment. The public and clean-tree trace artifacts have the same SHA256: `5741fbc7797cdb73b1def8222240961d7ddec39452b468cdd1c84168f8c5c70a`. The compact comparison record hash is `b2c1174033d3b7efbb31022f462f95cf0b7767fd37986f452f6dc459a1080fb5`. Those existing build records were preserved; the trace experiment was not rerun merely for a documentation change.

## Freeze and companion preservation

The [source-cutoff manifest](../provenance/SOURCE_CUTOFF.json) retains the validated snapshot from 2026-10-27 14:33–14:43 KST, frozen at 14:50:55 KST. Its recorded canonical digest is `02dcc4b2a1d6539df2ba6fc982f73acfecf9d0aba16117a5807aac5a735c5395`. The OMY history base is `985b9b0d82d4edce2d9bbf7dbfeee4919534924e`.

The source/reference revision changes documentation only. Controller, gain, coordinate mapping, DLS/null-space, conditioning, tests, launchers and configs remain byte-identical to the approved candidate. No original-workspace updates were imported.

The unchanged learning candidate is checked against its approved 343-file manifest instead of rerunning its tests. The earlier paired public and clean-tree suites each passed 723 tests with zero errors, failures and skips, including Square initialization/collection/data contracts. Those are retained baseline results, not a new 723-test run. Frontend build and result-regeneration evidence likewise remain the previously approved records.

## Outside the validation claim

- Historical OMY model loading, ROS bringup and exact scene/contact reproduction: not supplied or validated by this release.
- OpenHaptics SDK compilation and physical device acceptance: separate hardware-dependent checks.
- Physical Franka deployment, hard real-time behavior and human haptic comfort: not claimed.
- Fresh OS/driver/package-resolution installation: not established by the earlier relocated-tree check, which reused installed Python dependencies.
- Training or policy evaluation: not run in this revision. The companion's fixed results retain their documented provenance and reproduction limits.

Source/reference readiness does not remove these limits. See [historical asset findings](HISTORICAL_OMY_LIMITATIONS.md) and [scope](PUBLIC_V1_SCOPE.md).
