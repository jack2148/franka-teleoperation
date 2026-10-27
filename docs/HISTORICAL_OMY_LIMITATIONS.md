# Historical OMY asset provenance and reproduction limits

The historical OMY-L100 interface is retained as source/reference material. Its original MuJoCo scene/model bundle was not committed. Exact asset provenance could not be recovered, so recovery of the original executable environment is stopped.

## Evidence available

The public OMY history base is `985b9b0d82d4edce2d9bbf7dbfeee4919534924e`. Its preserved clone contains 39 reachable commits. Neither vendor model tree nor a submodule pin appears in the inspected reachable history; both model directories were ignored by Git.

The historical `simul` wrapper expects `mujoco_menagerie/franka_fr3/scene.xml` and checks for `fr3_front_wall`. The bridge also expects `robotis_mujoco_menagerie/robotis_omy/scene.xml`. The `e726c07` haptic-development commit introduces wall-consuming code, but does not preserve the wall XML, its dimensions, contact parameters or a vendor revision.

A 2026-09-04 migration record documents copying models from the original working tree and then removing the FR3 collision-test wall, wrist F/T sensors and haptic-only F/T site. It identifies the development revision as `e8de7be2de3ce35fb607d7d91c6481a02cbc3ba2`, which is absent from the inspected OMY clone object databases. The recorded bridge hash is not a model-bundle hash.

The resulting local model copies survive in a later Mimic import, `14b0b6cfc62d8ddec507f3f3a8a8bfbb95bcf6c5`. They establish the identity of that derivative, not the original native OMY executable environment. The OMY copy has a documented transfer path, but lacks a pinned native vendor revision/original asset hash. It cannot establish complete historical-bundle identity either.

## What is not reconstructed

The original wall/sensor/site XML and patch, exact native vendor revisions, and historical bundle hashes are unavailable. No modified Mimic copy, latest upstream scene or guessed patch is installed into the historical paths. No historical expected hash is invented.

Original launcher/config/controller files remain reference code. They do not constitute a ready-to-run reconstruction. Historical demos and development documents describe earlier observations, not a newly reproduced release.

## Validation boundary

CPU mathematical tests cover selected historical control functions. They do not establish original scene loading, contact behavior, ROS/OMY bringup or hardware operation. Geomagic CPU validation is a distinct interface scope, detailed in [VALIDATION.md](VALIDATION.md). Physical Franka deployment is not claimed.

This limitation is disclosed as part of the source/reference release contract. It is not hidden by replacing assets, skipping failing tests or presenting a different model as the historical one.
