# CDPL Server Smoke Record

## Context

- Local branch: `codex/cdpl-experiment`
- GitHub repository: `yangningning4186/CDPL`
- Verified commit on server: `0686f23`
- Server workspace: `/data4/ynz/CDPL-src`
- Conda environment: `ubt`
- GPU used for smoke: `CUDA_VISIBLE_DEVICES=0`

## Data Registration

The direct UBT/CDPL repository registers OCT datasets through environment variables:

- `oct_ss_train`: 607 images
- `oct_ss_train_unlabel`: 13541 images
- `oct_ss_test`: 478 images

Config checks confirmed:

- `SEMISUPNET.Trainer = cdpl`
- `MODEL.ROI_HEADS.NUM_CLASSES = 9`

## Verification Commands

Server compile check:

```bash
cd /data4/ynz/CDPL-src
/data4/ynz/anaconda3/bin/conda run -n ubt python -m py_compile \
  train_net.py \
  ubteacher/config.py \
  ubteacher/cdpl/*.py \
  ubteacher/engine/cdpl_trainer.py \
  ubteacher/engine/trainer.py \
  ubteacher/engine/writer_period.py \
  ubteacher/modeling/proposal_generator/rpn.py \
  ubteacher/data/datasets/oct_coco.py \
  ubteacher/data/datasets/builtin.py
```

Server unit check:

```bash
cd /data4/ynz/CDPL-src
/data4/ynz/anaconda3/bin/conda run -n ubt python -m unittest tests.ubteacher.test_writer_period
```

Result: `Ran 5 tests in 0.001s`, `OK`.

## Smoke Runs

### Warm-up Path

Purpose: verify the repository-native `train_net.py` entrypoint, OCT data loading, model build, supervised warm-up branch, checkpoint writing, and short-run hook compatibility.

Output:

- Run directory: `/data4/ynz/cdpl_runs/smoke_cdpl_src_burn2`
- Log file: `/data4/ynz/cdpl_runs/smoke_cdpl_src/burnup_2iter.log`
- Checkpoint: `/data4/ynz/cdpl_runs/smoke_cdpl_src_burn2/model_final.pth`

Result: command exited with status `0`.

### CDPL Branch Path

Purpose: verify `SEMISUPNET.Trainer=cdpl` can load the warm-up checkpoint and enter the post-burn-up CDPL/teacher-student path from the repository-native entrypoint.

Output:

- Run directory: `/data4/ynz/cdpl_runs/smoke_cdpl_src_cdpl2`
- Log file: `/data4/ynz/cdpl_runs/smoke_cdpl_src/cdpl_2iter.log`
- Checkpoint: `/data4/ynz/cdpl_runs/smoke_cdpl_src_cdpl2/model_final.pth`

Result: command exited with status `0`.

## Notes

- Do not use `/data4/ynz/OCT_SS-CDPL` for this experiment track. The verified path is `/data4/ynz/CDPL-src`.
- Do not use `SEMISUPNET.BURN_UP_STEP 0` with a randomly initialized teacher as a smoke substitute. Use a warm-up checkpoint before validating post-burn-up CDPL behavior.
- The server could not reliably pull from GitHub over HTTPS due TLS timeouts, so updates were synchronized with a Git bundle while preserving the same Git history and branch.
- A Detectron2 0.5 / PyTorch 1.8 CUDA indexing issue on RTX 3090 was avoided by sampling RPN labels on CPU inside `PseudoLabRPN._subsample_labels`.
