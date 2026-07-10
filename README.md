# NuRing / RingCellSeg

NuRing is a first-stage PyTorch implementation of nucleus-guided polar ring completion for whole-cell segmentation in multiplex immunofluorescence images.

The central assumption is that mIF marker signal often appears as incomplete perinuclear arcs instead of a closed membrane. NuRing therefore crops around each nucleus, converts the local image to polar space, predicts an angle-wise outer cell radius plus a polar cell mask, and converts the prediction back to a whole-cell instance mask.

## References Used For This Minimal Version

- Fu et al., TMI 2018: polar transformation and joint center/outer structure segmentation in optic disc/cup segmentation.
- Greenwald et al., Nature Biotechnology 2022: TissueNet/Mesmer task framing for nuclear and whole-cell segmentation in multiplexed tissue imaging.
- Cutler et al., Nature Methods 2022: Omnipose/Cellpose-style baseline awareness, distance-field intuition, and morphology-robust instance segmentation motivation.

CellSAM, MedSAM, topology consistency, shape-prior modules, and classification heads are intentionally left as future extensions.

## Project Structure

```text
configs/default.yaml
nuring/
  data/dataset.py
  data/polar_transform.py
  data/label_utils.py
  models/blocks.py
  models/polar_ring_net.py
  models/unet_baseline.py
  losses/losses.py
  baselines/dilation.py
  baselines/voronoi.py
  baselines/watershed.py
  postprocess/assignment.py
  postprocess/instance_utils.py
  metrics/segmentation_metrics.py
  metrics/radial_metrics.py
scripts/
  train_nuring.py
  infer_nuring.py
  run_baselines.py
  evaluate.py
  visualize_predictions.py
notebooks/demo_nuring.ipynb
```

## Data Layout

Default mIF paths are configured in `configs/default.yaml`:

```yaml
image_dir: /data/wzx/TLS/mIF50/registered
nucleus_dir: /data/wzx/TLS/mIF50/nucleiseg
tissuenet_dir: /data/wzx/tissuenet_v1.1
```

The minimal loader expects matched file stems:

```text
registered/sample001.npy      # [C,H,W] or [H,W,C]
nucleiseg/sample001.npy       # [H,W], nucleus instance ids, background 0
cells/sample001.npy optional  # [H,W], whole-cell instance ids
```

For the provided mIF50 data, `registered` is also supported in this directory form:

```text
/data/wzx/TLS/mIF50/registered/<sample>/registered_slides/DAPI.ome.tiff
/data/wzx/TLS/mIF50/registered/<sample>/registered_slides/CD20.ome.tiff
/data/wzx/TLS/mIF50/nucleiseg/<sample>.npy
```

The nucleus `.npy` loader supports both dense `[H,W]` masks and dict-style instance tables with `coords` and `label_id`. Reading OME-TIFF channels needs `tifffile` or Pillow; install `requirements.txt` in `wsi_img` if those are missing.

If `cell_dir` is absent, training uses a simple nucleus dilation pseudo-label. TissueNet can be exported into this layout with `export_tissuenet_to_npy()` in `nuring/data/dataset.py`.

## Train

```bash
conda activate wsi_img
cd /home/wzx/code/MTLab/NuRing
python scripts/train_nuring.py --config configs/default.yaml
```

Useful quick smoke test:

```bash
python scripts/train_nuring.py --config configs/default.yaml --epochs 1 --batch-size 2 --max-instances-per-image 16
```

Checkpoints are saved to `runs/nuring/best.pt` and `runs/nuring/last.pt`.

## Inference

```bash
python scripts/infer_nuring.py \
  --checkpoint runs/nuring/best.pt \
  --output-dir outputs/nuring \
  --threshold 0.5
```

Each output is saved as `outputs/nuring/<sample>_pred.npy`, shape `[H,W]`, with whole-cell instance ids.

## Baselines

```bash
python scripts/run_baselines.py --method dilation --radius 10 --output-dir outputs/baselines
python scripts/run_baselines.py --method voronoi --radius 32 --output-dir outputs/baselines
python scripts/run_baselines.py --method watershed --radius 32 --output-dir outputs/baselines
```

Implemented baselines:

- nucleus dilation with nearest-nucleus overlap resolution
- Voronoi-style nearest-nucleus assignment with max-radius cutoff
- watershed using nucleus seeds and marker-sum elevation when available

## Evaluation

```bash
python scripts/evaluate.py \
  --pred-dir outputs/nuring \
  --gt-dir /path/to/whole_cell_labels \
  --nucleus-dir /data/wzx/TLS/mIF50/nucleiseg \
  --output-csv outputs/eval_metrics.csv
```

Metrics currently include Dice, IoU, AJI, PQ, boundary F1, nucleus containment, and neighbor leakage.

## TissueNet Comparison And Ablation Experiments

Export a TissueNet subset into the `.npy` layout used by NuRing:

```bash
python scripts/export_tissuenet.py \
  --tissuenet-dir /data/wzx/tissuenet_v1.1 \
  --out-dir /data/wzx/tissuenet_v1.1/nuring_export_exp64 \
  --train-limit 64 --val-limit 16 --test-limit 16
```

Run baseline comparison and loss ablations:

```bash
python scripts/run_tissuenet_experiments.py \
  --export-dir /data/wzx/tissuenet_v1.1/nuring_export_exp64 \
  --work-dir runs/tissuenet_exp64 \
  --epochs 3 --batch-size 32 --num-workers 4 \
  --device cuda --base-channels 16 --max-instances-per-image 64
```

This runs:

- NuRing full loss
- mask-only ablation
- mask + radius without shape/containment/neighbor losses
- full loss without neighbor exclusion
- dilation, Voronoi, and watershed baselines

Summarize the results:

```bash
python scripts/summarize_experiments.py \
  --work-dir runs/tissuenet_exp64 \
  --train-count 64 --val-count 16 --test-count 16 --epochs 3
```

The experiment writes checkpoints, predictions, metric CSVs, `summary.json`, and `report.md` under `runs/tissuenet_exp64`.

## Visualization

```bash
python scripts/visualize_predictions.py \
  --image /data/wzx/TLS/mIF50/registered/sample001.npy \
  --nucleus /data/wzx/TLS/mIF50/nucleiseg/sample001.npy \
  --pred outputs/nuring/sample001_pred.npy \
  --baseline outputs/baselines/dilation/sample001_pred.npy \
  --out outputs/vis/sample001.png
```

## First-Stage Design Notes

- `cartesian_to_polar_tensor()` and `polar_to_cartesian_mask()` use `torch.nn.functional.grid_sample`, so the polar conversion can run on GPU.
- `PolarRingNet` consumes `[B,Cin,R,A]` and returns:
  - `radius`: `[B,A]`
  - `confidence`: `[B,A]`
  - `mask_logits`: `[B,1,R,A]`
- The radial head predicts a distribution over radial bins and uses expectation to produce a continuous radius.
- The loss combines polar mask Dice+BCE, confidence-weighted Smooth L1 radius loss, angular smoothness, nucleus containment, and neighbor nucleus exclusion.
- Global assignment resolves overlaps with `score = probability - 0.01 * distance_to_centroid`.

## Planned Extensions

1. CellSAM / MedSAM refinement using NuRing coarse masks or boxes as prompts.
2. Topology consistency loss and teacher-student semi-supervised training.
3. Graph cut or CRF global assignment.
4. Marker decontamination as preprocessing.
5. Few-shot fine-tuning with small manual annotations.
6. TLS crowded-region specific evaluation.
7. mIF marker overlap-aware segmentation.
8. Joint whole-cell segmentation and marker expression or cell type prediction.
