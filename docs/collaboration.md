# Working together

Use the main notebook and committed manifest for shared experiments. Each teammate can own one file in `src/eurosat/models/`: `resnet18.py`, `convnext_tiny.py`, `vit_b_16.py` or `dinov2_vits14.py`. Each adapter returns a pretrained model without its classification head; the shared loader freezes it and selects evaluation mode. Data selection, transforms, embedding normalization and classifier evaluation remain shared.

## Make a change

1. Get collaborator access or use a fork, then create a feature branch.
2. Edit the relevant source module. Discuss changes to the common data/evaluation protocol first.
3. Change notebook explanations/cells in `scripts/build_notebook.py` and regenerate only when those cells change.
4. Run a small smoke check and inspect the data audit, model dimensions and result tables.
5. Push the branch and open a pull request for teammate review.

```sh
python scripts/build_notebook.py
python scripts/execute_notebook.py --mode smoke --models vit_b_16 --output work/vit-smoke.ipynb
```

## Share artifacts

Use the shared Drive setup in [README](../README.md). Keep dataset archives, weights, embeddings and generated results outside Git. Use one writer per shared experiment at a time.

All-model runs automatically produce combined results. Matching caches can reuse work from individual model runs, although a different episode selection or chunk layout can require additional extraction. Compare only results with identical manifest, episode identities and evaluation settings. Share the experiment's `results.json` and provenance from Drive; regenerate its report without a GPU:

```sh
python -m eurosat report /path/to/results.json
```

Review full results before drawing conclusions. Smoke checks are not research results, and query accuracy must never choose settings.
