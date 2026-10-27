# Third-party and source provenance

The original OMY repository history base is `985b9b0d82d4edce2d9bbf7dbfeee4919534924e`. Existing project notices and history remain authoritative; no blanket license was added.

The author explicitly confirmed authorship and public distribution rights for the six Geomagic reference modules. Frozen copies are under `geomagic/vendor/geomagic_reference/`; their adapted runtime code is kept separately. The FR3 helper snapshot is also separate and is not an extracted common controller. The [cutoff manifest](provenance/SOURCE_CUTOFF.json) records file hashes.

OpenHaptics headers/runtime are externally licensed prerequisites and are not redistributed. C++ device interface and existing synthetic watchdog tests are included; the SDK-dependent callback test is in `hardware_validation/`. Physical haptic acceptance was not tested in this release check.

MuJoCo, ROS2 and OMY bringup are external dependencies. **The exact native OMY FR3/OMY scene assets and their pinned provenance are missing** from the source cutoff. The historical vendor trees were not committed; the remaining local FR3 derivative explicitly removed the front wall, wrist F/T sensors and F/T site. No replacement assets or guessed patches are included. See [historical asset limitations](docs/HISTORICAL_OMY_LIMITATIONS.md).

The approved contract is a **source/reference release**, not an OMY executable clean-clone reproduction. Missing historical asset provenance limits those reproduction claims; it does not imply a new asset license or authorization to reconstruct the original bundle. Existing source notices remain unchanged.

Square's MIT/Apache notices and environment provenance are in the paired learning repository. No investigation of excluded controller authorship is included in this audit.
