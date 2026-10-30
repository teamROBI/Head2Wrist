# Optional Transfer2.5 pilot

These small adapters prepare a Blender proxy render, choose a scene prompt and run the separately installed Cosmos-Transfer2.5 model. They are excluded from the default generator. The paper reports no improvement from this refinement at 20 demonstrations; the other large video-model pilot implementations are omitted.

Use `python -m baselines.blender_proxy --help`, `python -m baselines.caption_scene --help`, and `python -m baselines.run_transfer --help` in the corresponding environments after following `third_party/README.md`. Blender preparation uses its separate Python 3.13 environment and `PYTHONPATH` pointing to this repository. No model code is vendored.
