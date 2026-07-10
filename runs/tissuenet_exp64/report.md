# TissueNet NuRing Experiments

- TissueNet split: train=64, val=16, test=16
- NuRing epochs: 3
- Work dir: `runs/tissuenet_exp64`
- Metrics dir: `runs/tissuenet_exp64/metrics`
- Predictions dir: `runs/tissuenet_exp64/predictions`

| method | n | dice | iou | aji | pq | boundary_f1 | containment | leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nuring_full | 16.0000 | 0.7797 | 0.6568 | 0.4224 | 0.3199 | 0.2737 | 0.9969 | 0.0048 |
| ablate_mask_only | 16.0000 | 0.8856 | 0.8007 | 0.6032 | 0.5354 | 0.7232 | 0.9708 | 0.0106 |
| ablate_no_shape | 16.0000 | 0.7819 | 0.6592 | 0.4180 | 0.3088 | 0.2740 | 0.9950 | 0.0069 |
| ablate_no_neighbor | 16.0000 | 0.7827 | 0.6600 | 0.4234 | 0.3185 | 0.2726 | 0.9968 | 0.0049 |
| baseline_dilation | 16.0000 | 0.8210 | 0.7055 | 0.4822 | 0.3663 | 0.2880 | 1.0000 | 0.0000 |
| baseline_voronoi | 16.0000 | 0.7712 | 0.6486 | 0.4268 | 0.3408 | 0.2756 | 1.0000 | 0.0000 |
| baseline_watershed | 16.0000 | 0.7739 | 0.6513 | 0.3619 | 0.2853 | 0.2744 | 1.0000 | 0.0000 |

## Notes

- `nuring_full` uses mask, radius, smoothness, nucleus containment, and neighbor exclusion losses.
- `ablate_mask_only` disables radius, smoothness, containment, and neighbor losses.
- `ablate_no_shape` keeps mask + radius but disables smoothness, containment, and neighbor losses.
- `ablate_no_neighbor` disables only neighbor exclusion.
- Baselines use nucleus dilation, Voronoi nearest-nucleus assignment, and watershed.
