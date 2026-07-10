from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from nuring.utils.morphology import binary_dilation, binary_erosion


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * inter) / denom) if denom else 1.0


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 1.0


def _pairwise_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_ids = np.array([i for i in np.unique(pred) if i > 0], dtype=np.int32)
    gt_ids = np.array([i for i in np.unique(gt) if i > 0], dtype=np.int32)
    mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    pred_area = {int(i): int((pred == i).sum()) for i in pred_ids}
    gt_area = {int(i): int((gt == i).sum()) for i in gt_ids}
    for pi, pid in enumerate(pred_ids):
        p = pred == pid
        overlap_ids, counts = np.unique(gt[p], return_counts=True)
        for gid, inter in zip(overlap_ids, counts):
            if gid == 0:
                continue
            gj = np.where(gt_ids == gid)[0]
            if gj.size:
                union = pred_area[int(pid)] + gt_area[int(gid)] - int(inter)
                mat[pi, gj[0]] = float(inter) / max(union, 1)
    return pred_ids, gt_ids, mat


def aggregated_jaccard_index(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_ids, gt_ids, ious = _pairwise_iou(pred, gt)
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    used_pred = set()
    inter_sum = 0
    union_sum = 0
    for gj, gid in enumerate(gt_ids):
        if len(pred_ids) and ious[:, gj].max() > 0:
            pi = int(np.argmax(ious[:, gj]))
            used_pred.add(pi)
            p = pred == pred_ids[pi]
            g = gt == gid
            inter_sum += int(np.logical_and(p, g).sum())
            union_sum += int(np.logical_or(p, g).sum())
        else:
            union_sum += int((gt == gid).sum())
    for pi, pid in enumerate(pred_ids):
        if pi not in used_pred:
            union_sum += int((pred == pid).sum())
    return float(inter_sum / max(union_sum, 1))


def panoptic_quality(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> float:
    pred_ids, gt_ids, ious = _pairwise_iou(pred, gt)
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0
    matched_pred = set()
    matched_gt = set()
    iou_sum = 0.0
    pairs = [(float(ious[i, j]), i, j) for i in range(ious.shape[0]) for j in range(ious.shape[1]) if ious[i, j] >= iou_thr]
    for iou, pi, gj in sorted(pairs, reverse=True):
        if pi in matched_pred or gj in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gj)
        iou_sum += iou
    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return float(iou_sum / max(tp + 0.5 * fp + 0.5 * fn, 1e-6))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    pred_b = (pred > 0) ^ binary_erosion(pred > 0)
    gt_b = (gt > 0) ^ binary_erosion(gt > 0)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    pred_match = pred_b & binary_dilation(gt_b, iterations=tolerance)
    gt_match = gt_b & binary_dilation(pred_b, iterations=tolerance)
    precision = pred_match.sum() / max(pred_b.sum(), 1)
    recall = gt_match.sum() / max(gt_b.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-6))


def nucleus_containment_rate(pred: np.ndarray, nuclei: np.ndarray) -> float:
    rates = []
    for nid in np.unique(nuclei):
        if nid == 0:
            continue
        nuc = nuclei == nid
        rates.append(float(np.mean(pred[nuc] == nid)))
    return float(np.mean(rates)) if rates else 1.0


def neighbor_leakage_rate(pred: np.ndarray, nuclei: np.ndarray) -> float:
    leaks = []
    for pid in np.unique(pred):
        if pid == 0:
            continue
        region = pred == pid
        other_nuc = (nuclei > 0) & (nuclei != pid)
        leaks.append(float(np.logical_and(region, other_nuc).sum() / max(region.sum(), 1)))
    return float(np.mean(leaks)) if leaks else 0.0


def compute_instance_metrics(pred: np.ndarray, gt: np.ndarray, nuclei: np.ndarray | None = None) -> Dict[str, float]:
    metrics = {
        "dice": binary_dice(pred, gt),
        "iou": binary_iou(pred, gt),
        "aji": aggregated_jaccard_index(pred, gt),
        "pq": panoptic_quality(pred, gt),
        "boundary_f1": boundary_f1(pred, gt),
    }
    if nuclei is not None:
        metrics["nucleus_containment"] = nucleus_containment_rate(pred, nuclei)
        metrics["neighbor_leakage"] = neighbor_leakage_rate(pred, nuclei)
    return metrics
