# Context-Aware Student Engagement Fine-Tuning

This is a lightweight fine-tuning workspace for adding `Sleeping` and
`Talking` to the existing 13 action labels.

## Included data

`data_split/` contains local copies of the videos. The original datasets remain
unchanged. The split is deterministic with seed 42:

- train: 35 videos per new class
- val: 10 videos per new class
- test: 5 videos per new class

The annotation files under `annotations/` use the ViFiCLIP format:
`relative_video_path label_id`.

## Model files

The ViFiCLIP source is under `ViFiCLIP/`. The existing checkpoint is referenced
by a symlink at
`pretrained_action_recog_models/ckpt_epoch_30.pth`; it is not copied into this
workspace.

Install dependencies from the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The adapted configuration is
`pretrained_action_recog_models/16_32_vifi_clip_16_shot_cvpr_n_actions.yaml`.
It defines 15 classes and uses the local annotation files.

## Dry run and training

Run a dependency/model/data smoke test without updating weights:

```bash
python finetune.py --dry-run --device cpu
```

Start fine-tuning:

```bash
python finetune.py --device cuda --resume \
  pretrained_action_recog_models/ckpt_epoch_30.pth
```

Outputs are written to `outputs/best_lora.pth`. Only LoRA adapter weights are
saved; the source checkpoint is shared by symlink, so it is not duplicated.
The current split includes seven labels with available videos. Training only
those labels can reduce performance on the other eight labels.

Evaluate only the 10 held-out `Sleeping` and `Talking` videos:

```bash
python evaluate_new_classes.py --device cuda \
  --adapter outputs/best_lora.pth
```

The predictions are written to
`outputs/new_classes_predictions.csv`. The base checkpoint is loaded through
the existing symlink and the small LoRA adapter is loaded from `--adapter`.

Compare the original and LoRA models quantitatively on all 35 held-out test
videos:

```bash
python evaluate_comparison.py --device cuda
```

Results are written under `outputs/full_comparison/`:

- `base_predictions.csv` and `lora_predictions.csv`
- `base_per_class.csv` and `lora_per_class.csv`
- confusion matrices
- `comparison_summary.json` with accuracy and macro-F1 changes
