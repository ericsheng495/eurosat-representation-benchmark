# Dataset checks and inspection

EuroSAT RGB contains 27,000 native 64×64 RGB tiles across ten classes. This checkout's decoded-pixel audit found zero exact duplicates and zero corrupt images. The benchmark evaluates the four novel classes defined in the [committed manifest](../SPLITS.md); the six base classes are reserved.

Notebook section 3 exposes the processing steps:

1. **Verify source:** check the persistent ZIP checksum, check extracted files against archive CRCs, restore missing/changed files, and audit decoded pixels. Display the ten-class source inventory and native image properties.
2. **Validate manifest:** check complete pool coverage, filenames, class membership, partition isolation and every support/query episode. Display pool sizes and episode counts.
3. **Select assignments:** choose the fixed smoke/full episode prefix, export records and assignments, and show actual image paths and reuse counts.
4. **Inspect inputs:** display native development support images and novel pool counts, then check normalized tensors for shape, finite values and repeatable transforms.

Audit JSON files, image assignments, usage CSV and generated PNG/PDF figures live under `cache/data-audits/<manifest-id>/<selection-id>/`. Follow the notebook's printed path to inspect them. Examples come from development supports; test queries do not select settings or illustrative examples.

The novel pools contain 3,000 development and 7,000 test images. Each episode samples four classes with 15 query images per class; images may recur across episodes. This separation does not establish geographic independence.

Attribution: Patrick Helber, Benjamin Bischke, Andreas Dengel and Damian Borth, “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,” IEEE JSTARS, 2019. Sentinel-2 imagery: Copernicus. See the [official dataset repository](https://github.com/phelber/EuroSAT) for citation and terms.
