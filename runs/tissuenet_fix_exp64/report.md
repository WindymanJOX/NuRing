# TissueNet NuRing Fixed-Loop Smoke Test

- TissueNet split: train=64, val=16, test=16
- NuRing epochs: 3
- Experiment kind: smoke test; use this to validate code paths, not as a final method conclusion.
- Work dir: `runs/tissuenet_fix_exp64`
- Metrics dir: `runs/tissuenet_fix_exp64/metrics`
- Predictions dir: `runs/tissuenet_fix_exp64/predictions`

| method | n | dice | iou | aji | pq | boundary_f1 | containment | leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss_mask_only | 16.0000 | 0.8813 | 0.7971 | 0.5968 | 0.5268 | 0.7054 | 0.9922 | 0.0123 |
| loss_mask_radius | 16.0000 | 0.7758 | 0.6530 | 0.4341 | 0.3430 | 0.2740 | 0.9906 | 0.0121 |
| loss_mask_radius_conf | 16.0000 | 0.7758 | 0.6531 | 0.4344 | 0.3413 | 0.2740 | 0.9880 | 0.0144 |
| loss_mask_radius_conf_smooth | 16.0000 | 0.7758 | 0.6530 | 0.4341 | 0.3419 | 0.2740 | 0.9881 | 0.0143 |
| nuring_full | 16.0000 | 0.7757 | 0.6529 | 0.4340 | 0.3419 | 0.2740 | 0.9912 | 0.0108 |
| fusion_mask_only | 16.0000 | 0.8820 | 0.7981 | 0.5949 | 0.5246 | 0.7035 | 0.9941 | 0.0097 |
| fusion_mask_radius_average | 16.0000 | 0.7757 | 0.6529 | 0.4339 | 0.3430 | 0.2740 | 0.9925 | 0.0097 |
| fusion_confidence_fusion | 16.0000 | 0.7757 | 0.6529 | 0.4340 | 0.3419 | 0.2740 | 0.9912 | 0.0108 |
| baseline_dilation | 16.0000 | 0.8210 | 0.7055 | 0.4822 | 0.3663 | 0.2880 | 1.0000 | 0.0000 |
| baseline_voronoi | 16.0000 | 0.7712 | 0.6486 | 0.4268 | 0.3408 | 0.2756 | 1.0000 | 0.0000 |
| baseline_watershed | 16.0000 | 0.7739 | 0.6513 | 0.3619 | 0.2853 | 0.2744 | 1.0000 | 0.0000 |

## Notes

- `loss_mask_only` trains and infers with only the polar mask branch.
- `loss_mask_radius` adds normalized radius supervision and uses mask/radius average at inference.
- `loss_mask_radius_conf` adds confidence supervision and uses confidence fusion.
- `loss_mask_radius_conf_smooth` adds normalized circular smoothness.
- `nuring_full` adds containment and neighbor exclusion with reduced default weights.
- `fusion_mask_only`, `fusion_mask_radius_average`, and `fusion_confidence_fusion` evaluate the same full checkpoint with different inference fusion modes.
- Baselines use nucleus dilation, Voronoi nearest-nucleus assignment, and watershed.
