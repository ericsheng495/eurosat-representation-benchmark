# Four-person collaboration

Agree on owners before experiments. These are suggested workstreams, not assigned names; the proposal authors are Andrew Dao, Eric Sheng, Eunhyeok Jang and Yanru Yang.

| Notebook part | Suggested owner / reviewer | Backbone assignment | Shared responsibility |
|---|---|---|---|
| A | Teammate 1 / 4 | `resnet18` | Data audit, partitions, baseline, reproducibility |
| B | Teammate 2 / 1 | `convnext_tiny` | Geometric policies, feature caching, CNN comparisons |
| C | Teammate 3 / 2 | `vit_b_16` | Color/pixel policies, preprocessing, Transformer comparisons |
| D | Teammate 4 / 3 | `dinov2_vits14` | Metrics, merged exports, figures and paper integration |

Every backbone owner runs **every shared policy**, including controls. Policy ownership means implementation/review, not different experimental grids. Use notebook `ASSIGNMENT` to select your backbone. The Part A–D configuration cells are preparation only; the final execution cell runs the selected `config` once.

## Daily workflow

1. Clone the agreed GitHub repository and follow the README setup. Claim a TODO item with your name and branch in the team tracker before working.
2. Update main, then create a focused branch: `git switch -c feat/short-description`.
3. Edit the linked source functions, not generated embedded notebook helpers. Add meaningful checks when behavior changes. Rebuild the notebook with `python scripts/build_notebook.py`.
4. Run `python -m unittest discover -s tests -v`. For model/runtime changes also execute a real smoke notebook: `python scripts/execute_notebook.py --models YOUR_MODEL`.
5. Commit source, tests, notebook and relevant documentation. Open a pull request with the change, validation and any protocol impact. Ask the listed reviewer to review before merging. Update checkboxes in TODO.md and the notebook generator together.
6. Pull merged changes before starting the next item. Never force-push a teammate's branch or main.

Recommended repository settings: require a passing CI check and one teammate review on main. Agree on shared protocol changes with all four owners before long runs. These settings require repository configuration and are not enabled merely by this document.

## Shared protocol and Drive

Freeze the seed, dataset identity, checkpoint IDs, software versions, original shots, view counts, strengths, classifier grid and trial count before comparison. Test labels must not choose policies or strengths. Record deviations before interpreting results. Development labels are additional labeled selection data and must be disclosed.

Each account's `MyDrive` is separate. To share storage, one owner grants the other three access to a Drive folder; each teammate adds a shortcut and sets the notebook `STORAGE` to their accessible path. Identical path strings alone do not grant access. Dataset archives and caches may be shared; each runtime keeps its own scratch extraction directory.

Assign distinct backbones to avoid simultaneous writers for the same experiment. Never manually edit cached NPZ/JSON files. Atomic writes protect interruptions, but do not coordinate identical concurrent jobs. Changes to pipeline semantics require a reviewed `PIPELINE_VERSION` increment. Keep credentials, datasets, weights, caches and full execution outputs out of Git.

## Combine experiments

Each owner exports their completed `outputs/<experiment-id>/results.json`. Copy the four reports to an agreed results directory, then:

```sh
python -m eurosat merge reports/resnet.json reports/convnext.json reports/vit.json reports/dino.json --output outputs/combined
```

The merger rejects different data, software, shared configuration or trial manifests, missing trials, and duplicate experimental rows. Model subsets may differ. Keep original exports for provenance. Share compact reviewed figures/tables with the paper; preserve raw reports in shared storage. Confirm all four backbone assignments are present before claiming a complete comparison.
