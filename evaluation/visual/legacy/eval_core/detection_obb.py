import re
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def process_flat_obb(a):
    """Convert 8-number list [x1,y1,x2,y2,x3,y3,x4,y4] to a Shapely polygon (or False if invalid)."""
    if len(a) != 8:
        return False
    poly = Polygon([(a[0], a[1]), (a[2], a[3]), (a[4], a[5]), (a[6], a[7])])
    return poly if poly.is_valid else False

def extract_class(text):
    patterns = [
        r"Detect all ([\w\s]+?) in the image",
        r"Segment out ([\w\s]+?) in the image",
        r"Is there a ([\w\s]+?) in the image"
    ]
    if not isinstance(text, str):
        return None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

def _parse_numbers(s):
    """Parse floats (including negatives/scientific) from a string."""
    return [float(x) for x in _NUM_RE.findall(s)]

def format_preprocess(data, is_box=False, is_obb=False, isGT=False):
    """
    Original behavior preserved:
    - If is_box=True: parse <quad>...</quad> or <box>...</box> tags inside a string.
    - Else: if data is list, cast to float lists; else return [].
    """
    if is_box:
        if not isinstance(data, str):
            return []
        tag = '<quad>' if is_obb else '<box>'
        end_tag = tag.replace("<", "</")
        boxes = re.findall(f'{tag}(.*?){end_tag}', data)
        return [_parse_numbers(box) for box in boxes]
    return [[float(a) for a in x] for x in data] if isinstance(data, list) else []

def calculate_iou_obb(poly1, poly2):
    if not poly1 or not poly2 or not poly1.intersects(poly2):
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

def calculate_iou_hbb(box1, box2):
    try:
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
        inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area + 0.1
        return inter_area / union_area
    except Exception:
        return 1

def calculate_ap(precision, recall):
    """
    Kept for backward compatibility.
    """
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    indices = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

def _best_matches_obb(gt_polys, pred_polys):
    """
    For each pred polygon, find (best_iou, best_gt_index) among GT polygons.
    Uses STRtree when GT count is large.
    Returns:
      best_iou: float32 array (len=preds)
      best_idx: int32 array (len=preds), index into ORIGINAL gt list, or -1
    """
    n_pred = len(pred_polys)
    best_iou = np.zeros(n_pred, dtype=np.float32)
    best_idx = np.full(n_pred, -1, dtype=np.int32)

    # Keep GT indexing consistent with the original list
    valid_gt = [(i, g) for i, g in enumerate(gt_polys) if g]
    if not valid_gt or n_pred == 0:
        return best_iou, best_idx

    gt_geoms = [g for _, g in valid_gt]
    gt_orig_idx = np.array([i for i, _ in valid_gt], dtype=np.int32)
    gt_area = np.array([g.area for g in gt_geoms], dtype=np.float32)

    use_tree = (len(gt_geoms) >= 20) and (n_pred >= 20)
    if use_tree:
        tree = STRtree(gt_geoms)
        # Map geometry object -> index in gt_geoms
        geom_to_k = {id(g): k for k, g in enumerate(gt_geoms)}
    else:
        tree = None
        geom_to_k = None

    for pi, p in enumerate(pred_polys):
        if not p:
            continue

        p_area = float(p.area)
        # Candidate pruning by bbox via STRtree (when enabled)
        if tree is not None:
            candidates = tree.query(p)
            candidate_iter = candidates
        else:
            candidate_iter = gt_geoms

        bi = 0.0
        bj = -1

        if tree is None:
            for k, g in enumerate(candidate_iter):
                try:
                    if not p.intersects(g):
                        continue
                    inter = p.intersection(g).area
                except Exception:
                    continue
                if inter <= 0.0:
                    continue
                union = p_area + float(gt_area[k]) - inter
                if union <= 0.0:
                    continue
                iou = inter / union
                if iou > bi:
                    bi = iou
                    bj = int(gt_orig_idx[k])
        else:
            for g in candidate_iter:
                k = geom_to_k.get(id(g), None)
                if k is None:
                    continue
                try:
                    if not p.intersects(g):
                        continue
                    inter = p.intersection(g).area
                except Exception:
                    continue
                if inter <= 0.0:
                    continue
                union = p_area + float(gt_area[k]) - inter
                if union <= 0.0:
                    continue
                iou = inter / union
                if iou > bi:
                    bi = iou
                    bj = int(gt_orig_idx[k])

        best_iou[pi] = bi
        best_idx[pi] = bj

    return best_iou, best_idx

def _best_matches_hbb(gt_boxes, pred_boxes):
    n_pred = len(pred_boxes)
    best_iou = np.zeros(n_pred, dtype=np.float32)
    best_idx = np.full(n_pred, -1, dtype=np.int32)

    if len(gt_boxes) == 0 or n_pred == 0:
        return best_iou, best_idx

    for i, pb in enumerate(pred_boxes):
        bi = 0.0
        bj = -1
        for j, gb in enumerate(gt_boxes):
            iou = calculate_iou_hbb(pb, gb)
            if iou > bi:
                bi = iou
                bj = j
        best_iou[i] = bi
        best_idx[i] = bj
    return best_iou, best_idx

def _ap_from_tp_ranks(tp_ranks_sorted, total_gts):
    """
    Compute AP exactly like calculate_ap() would, but using only TP positions.
    tp_ranks_sorted: 0-based positions in the sorted-by-score list, ascending.
    total_gts: total GT count.
    Returns AP in [0,1].
    """
    if total_gts <= 0:
        return 0.0
    m = int(tp_ranks_sorted.size)
    if m == 0:
        return 0.0

    # At TP k, precision = k / (rank_k + 1)
    k = np.arange(1, m + 1, dtype=np.float64)
    ranks1 = tp_ranks_sorted.astype(np.float64) + 1.0
    prec = k / ranks1

    # Precision envelope (suffix max)
    prec_env = np.maximum.accumulate(prec[::-1])[::-1]

    # Recall increases by 1/total_gts at each TP
    return float(prec_env.sum() / float(total_gts))

def _safe_class_name(sample):
    try:
        c = extract_class(sample.get('question', None))
        if c:
            return c.replace(" ", "_")
    except Exception:
        pass
    return "unknown"

def evaluate_detection2(data, pre_box=False, is_box=False, is_obb=False):
    """
    Optimized multi-threshold evaluation. Output keys preserved:
      Precision@t, Recall@t, AP@t, Accuracy@t, FAR@t, F1@t,
      class_Precision@t, class_Recall@t, class_AP@t, class_Accuracy@t, class_FAR@t,
      gt_per_class_AP@t
    """
    # Thresholds 0.1..0.9
    thresholds = [round(0.1 * i, 1) for i in range(1, 10)]
    re_dict = {}

    # ---- Precompute per-sample best IoUs ONCE ----
    gt_counter_per_class = {}
    class_iou_chunks = {}
    class_chosen_idx_chunks = {}
    class_chosen_iou_chunks = {}
    class_num_gts = {}
    class_num_preds = {}

    all_iou_chunks = []
    all_chosen_idx_chunks = []
    all_chosen_iou_chunks = []

    overall_pred_offset = 0
    class_pred_offset = {}

    # We still cache parsed_gt/parsed_answer like the original file (small win on repeated runs)
    for sample in tqdm(data, desc="Precomputing IoUs", leave=False):
        class_name = _safe_class_name(sample)
        gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1

        if 'parsed_gt' not in sample:
            sample['parsed_gt'] = format_preprocess(sample.get('gt', []), is_box, is_obb, isGT=True)
        if 'parsed_answer' not in sample:
            sample['parsed_answer'] = format_preprocess(sample.get('answer', []), pre_box, is_obb, isGT=False)

        gt_boxes = sample['parsed_gt']
        pred_boxes = sample['parsed_answer']

        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)

        class_num_gts[class_name] = class_num_gts.get(class_name, 0) + gt_count
        class_num_preds[class_name] = class_num_preds.get(class_name, 0) + pred_count

        # Ensure per-class containers exist
        if class_name not in class_iou_chunks:
            class_iou_chunks[class_name] = []
            class_chosen_idx_chunks[class_name] = []
            class_chosen_iou_chunks[class_name] = []
            class_pred_offset[class_name] = 0

        # Record offsets BEFORE consuming this sample
        this_overall_offset = overall_pred_offset
        this_class_offset = class_pred_offset[class_name]

        if pred_count == 0:
            empty = np.empty((0,), dtype=np.float32)
            class_iou_chunks[class_name].append(empty)
            all_iou_chunks.append(empty)
            # offsets update
            # (no preds, so offsets stay)
            continue

        if is_obb:
            gt_polys = [process_flat_obb(b) for b in gt_boxes]
            pred_polys = [process_flat_obb(b) for b in pred_boxes]
            best_iou, best_idx = _best_matches_obb(gt_polys, pred_polys)
        else:
            best_iou, best_idx = _best_matches_hbb(gt_boxes, pred_boxes)

        # Best pred per GT (argmax IoU among preds whose best match is that GT)
        best_pred_for_gt = np.full(gt_count, -1, dtype=np.int32)
        best_pred_iou_for_gt = np.full(gt_count, -1.0, dtype=np.float32)

        for pi in range(pred_count):
            gi = int(best_idx[pi])
            if gi >= 0 and gi < gt_count:
                bi = float(best_iou[pi])
                if bi > float(best_pred_iou_for_gt[gi]):
                    best_pred_iou_for_gt[gi] = bi
                    best_pred_for_gt[gi] = pi

        chosen_mask = best_pred_for_gt >= 0
        chosen_local_idx = best_pred_for_gt[chosen_mask].astype(np.int32, copy=False)
        chosen_iou = best_pred_iou_for_gt[chosen_mask].astype(np.float32, copy=False)

        if chosen_local_idx.size:
            all_chosen_idx_chunks.append((this_overall_offset + chosen_local_idx).astype(np.int32, copy=False))
            all_chosen_iou_chunks.append(chosen_iou)

            class_chosen_idx_chunks[class_name].append((this_class_offset + chosen_local_idx).astype(np.int32, copy=False))
            class_chosen_iou_chunks[class_name].append(chosen_iou)

        # Accumulate IoUs for sorting/ranks
        class_iou_chunks[class_name].append(best_iou.astype(np.float32, copy=False))
        all_iou_chunks.append(best_iou.astype(np.float32, copy=False))

        # Update offsets AFTER
        overall_pred_offset += pred_count
        class_pred_offset[class_name] += pred_count

    total_preds = overall_pred_offset
    total_gts = int(sum(class_num_gts.values()))

    # ---- Build concatenated IoU arrays + rank maps (sorted once) ----
    if all_iou_chunks:
        all_iou = np.concatenate(all_iou_chunks, axis=0) if total_preds > 0 else np.empty((0,), dtype=np.float32)
    else:
        all_iou = np.empty((0,), dtype=np.float32)

    if all_chosen_idx_chunks:
        all_chosen_idx = np.concatenate(all_chosen_idx_chunks, axis=0)
        all_chosen_iou = np.concatenate(all_chosen_iou_chunks, axis=0)
    else:
        all_chosen_idx = np.empty((0,), dtype=np.int32)
        all_chosen_iou = np.empty((0,), dtype=np.float32)

    if total_preds > 0:
        # ranks: position in descending-IoU order for each pred index
        sorted_idx = np.argsort(-all_iou, kind='mergesort')
        all_rank = np.empty(total_preds, dtype=np.int32)
        all_rank[sorted_idx] = np.arange(total_preds, dtype=np.int32)
        del sorted_idx
    else:
        all_rank = np.empty((0,), dtype=np.int32)

    # Per-class ranks and chosen arrays
    class_rank = {}
    class_iou = {}
    class_chosen_idx = {}
    class_chosen_iou = {}

    for cname, chunks in class_iou_chunks.items():
        n_pred = int(class_num_preds.get(cname, 0))
        if n_pred > 0:
            ciou = np.concatenate(chunks, axis=0)
            sidx = np.argsort(-ciou, kind='mergesort')
            crank = np.empty(n_pred, dtype=np.int32)
            crank[sidx] = np.arange(n_pred, dtype=np.int32)
            del sidx
        else:
            ciou = np.empty((0,), dtype=np.float32)
            crank = np.empty((0,), dtype=np.int32)

        if class_chosen_idx_chunks.get(cname) and sum(x.size for x in class_chosen_idx_chunks[cname]) > 0:
            cidx = np.concatenate(class_chosen_idx_chunks[cname], axis=0)
            ciou_ch = np.concatenate(class_chosen_iou_chunks[cname], axis=0)
        else:
            cidx = np.empty((0,), dtype=np.int32)
            ciou_ch = np.empty((0,), dtype=np.float32)

        class_iou[cname] = ciou
        class_rank[cname] = crank
        class_chosen_idx[cname] = cidx
        class_chosen_iou[cname] = ciou_ch

    # ---- Compute metrics for each threshold ----
    for thr in tqdm(thresholds, desc="Processing IoUs"):
        # Overall TP count is #chosen_iou >= thr
        if all_chosen_iou.size:
            mask = all_chosen_iou >= thr
            tp_pred_idx = all_chosen_idx[mask]
            tp_count = int(tp_pred_idx.size)
            tp_ranks = np.sort(all_rank[tp_pred_idx]) if tp_count else np.empty((0,), dtype=np.int32)
        else:
            tp_count = 0
            tp_ranks = np.empty((0,), dtype=np.int32)

        p = (tp_count / total_preds) if total_preds > 0 else 0.0
        r = (tp_count / total_gts) if total_gts > 0 else 0.0
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        ap = _ap_from_tp_ranks(tp_ranks, total_gts) if total_gts > 0 else 0.0
        denom = (total_preds + total_gts - tp_count)
        accuracy = (tp_count / denom) if denom > 0 else 0.0
        far = ((total_preds - tp_count) / denom) if denom > 0 else 0.0

        re_dict[f'Precision@{thr}'] = p * 100.0
        re_dict[f'Recall@{thr}'] = r * 100.0
        re_dict[f'AP@{thr}'] = ap * 100.0
        re_dict[f'Accuracy@{thr}'] = accuracy * 100.0
        re_dict[f'FAR@{thr}'] = far * 100.0
        re_dict[f'F1@{thr}'] = f1 * 100.0

        # Per-class metrics
        class_precision = {}
        class_recall = {}
        class_ap = {}
        class_accuracy = {}
        class_far = {}

        for cname in class_num_preds.keys():
            n_pred = int(class_num_preds.get(cname, 0))
            n_gt = int(class_num_gts.get(cname, 0))

            if n_pred <= 0 or n_gt < 0:
                class_precision[cname] = 0
                class_recall[cname] = 0
                class_ap[cname] = 0
                class_accuracy[cname] = 0
                class_far[cname] = 0
                continue

            c_ch_iou = class_chosen_iou.get(cname, np.empty((0,), dtype=np.float32))
            c_ch_idx = class_chosen_idx.get(cname, np.empty((0,), dtype=np.int32))
            c_rank = class_rank.get(cname, np.empty((0,), dtype=np.int32))

            if c_ch_iou.size:
                cmask = c_ch_iou >= thr
                ctp_idx = c_ch_idx[cmask]
                ctp_count = int(ctp_idx.size)
                ctp_ranks = np.sort(c_rank[ctp_idx]) if ctp_count else np.empty((0,), dtype=np.int32)
            else:
                ctp_count = 0
                ctp_ranks = np.empty((0,), dtype=np.int32)

            cp = (ctp_count / n_pred) if n_pred > 0 else 0.0
            cr = (ctp_count / n_gt) if n_gt > 0 else 0.0
            cap = _ap_from_tp_ranks(ctp_ranks, n_gt) if n_gt > 0 else 0.0

            cden = (n_pred + n_gt - ctp_count)
            cacc = (ctp_count / cden) if cden > 0 else 0.0
            cfar = ((n_pred - ctp_count) / cden) if cden > 0 else 0.0

            class_precision[cname] = cp * 100.0
            class_recall[cname] = cr * 100.0
            class_ap[cname] = cap * 100.0
            class_accuracy[cname] = cacc * 100.0
            class_far[cname] = cfar * 100.0

        re_dict[f'class_Precision@{thr}'] = class_precision
        re_dict[f'class_Recall@{thr}'] = class_recall
        re_dict[f'class_AP@{thr}'] = class_ap
        re_dict[f'class_Accuracy@{thr}'] = class_accuracy
        re_dict[f'class_FAR@{thr}'] = class_far

        # Original code stores this same dict for every threshold
        re_dict[f'gt_per_class_AP@{thr}'] = gt_counter_per_class

    return re_dict

# Backward-compatible wrappers (optional use)
def evaluate_one(gt_boxes, pred_boxes, iou_threshold=0.1, is_obb=False):
    """
    Kept for compatibility, now without joblib.Parallel.
    Returns the same tuple as the original evaluate_one.
    """
    gt_polys = [process_flat_obb(box) if is_obb else box for box in gt_boxes]
    pred_polys = [process_flat_obb(box) if is_obb else box for box in pred_boxes]

    if is_obb:
        ious, best_idx = _best_matches_obb(gt_polys, pred_polys)
    else:
        ious, best_idx = _best_matches_hbb(gt_boxes, pred_boxes)

    match_ious = np.array([int(best_idx[i]) if ious[i] >= iou_threshold else -1 for i in range(len(pred_polys))], dtype=np.int32)

    refune_ious = np.zeros(len(pred_polys), dtype=np.float32)
    tp = np.zeros(len(pred_polys), dtype=np.float32)
    fp = np.ones(len(pred_polys), dtype=np.float32)
    detected_gt = set()

    for j in range(len(gt_boxes)):
        matching_preds = np.where(match_ious == j)[0]
        if matching_preds.size:
            best_pi = matching_preds[np.argmax(ious[matching_preds])]
            refune_ious[best_pi] = float(ious[best_pi])
            tp[best_pi], fp[best_pi] = 1.0, 0.0
            detected_gt.add(j)

    fn = len(gt_boxes) - len(detected_gt)
    return tp, fp, ious, refune_ious, fn, len(pred_boxes), len(gt_boxes)

def calculate_eval_matrix(data, iou_threshold=0.1, pred_format=False, gt_format=False, is_obb=False):
    """
    Wrapper for a single threshold (kept compatible).
    This still computes IoUs per call; evaluate_detection2 is the optimized multi-threshold path.
    """
    total_tp_chunks, total_fp_chunks, total_iou_chunks = [], [], []
    total_fn, total_pres, total_gts = 0, 0, 0
    class_tp_chunks, class_fp_chunks, class_iou_chunks = {}, {}, {}
    class_gt, class_total_fn, class_total_pres = {}, {}, {}
    gt_counter_per_class = {}

    for a in data:
        class_name = _safe_class_name(a)
        gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1

        if 'parsed_gt' not in a:
            a['parsed_gt'] = format_preprocess(a.get('gt', []), gt_format, is_obb, isGT=True)
        if 'parsed_answer' not in a:
            a['parsed_answer'] = format_preprocess(a.get('answer', []), pred_format, is_obb, isGT=False)

        tp, fp, iou, _, fn, pres, gts = evaluate_one(a['parsed_gt'], a['parsed_answer'], iou_threshold, is_obb)

        total_tp_chunks.append(tp.astype(np.float32, copy=False))
        total_fp_chunks.append(fp.astype(np.float32, copy=False))
        total_iou_chunks.append(iou.astype(np.float32, copy=False))
        total_fn += int(fn)
        total_pres += int(pres)
        total_gts += int(gts)

        class_tp_chunks.setdefault(class_name, []).append(tp.astype(np.float32, copy=False))
        class_fp_chunks.setdefault(class_name, []).append(fp.astype(np.float32, copy=False))
        class_iou_chunks.setdefault(class_name, []).append(iou.astype(np.float32, copy=False))
        class_gt[class_name] = class_gt.get(class_name, 0) + int(gts)
        class_total_fn[class_name] = class_total_fn.get(class_name, 0) + int(fn)
        class_total_pres[class_name] = class_total_pres.get(class_name, 0) + int(pres)

    class_precision, class_recall, class_AP = {}, {}, {}
    class_accuracy, class_far = {}, {}

    for class_name in class_tp_chunks.keys():
        tp_arr = np.concatenate(class_tp_chunks[class_name], axis=0) if class_tp_chunks[class_name] else np.empty((0,), dtype=np.float32)
        fp_arr = np.concatenate(class_fp_chunks[class_name], axis=0) if class_fp_chunks[class_name] else np.empty((0,), dtype=np.float32)
        iou_arr = np.concatenate(class_iou_chunks[class_name], axis=0) if class_iou_chunks[class_name] else np.empty((0,), dtype=np.float32)

        sorted_idx = np.argsort(-iou_arr) if iou_arr.size else np.empty((0,), dtype=np.int64)
        tp_cum = np.cumsum(tp_arr[sorted_idx]) if sorted_idx.size else np.array([], dtype=np.float32)
        fp_cum = np.cumsum(fp_arr[sorted_idx]) if sorted_idx.size else np.array([], dtype=np.float32)

        recall = tp_cum / float(class_gt[class_name]) if class_gt[class_name] > 0 else np.zeros(len(tp_cum), dtype=np.float32)
        precision = tp_cum / np.arange(1, len(tp_cum) + 1, dtype=np.float32) if len(tp_cum) else np.array([], dtype=np.float32)

        class_recall[class_name] = float(recall[-1] * 100) if len(recall) else 0.0
        class_precision[class_name] = float(precision[-1] * 100) if len(precision) else 0.0
        class_AP[class_name] = float(calculate_ap(precision, recall) * 100) if len(precision) else 0.0

        tp_final = float(tp_cum[-1]) if len(tp_cum) else 0.0
        fp_final = float(fp_cum[-1]) if len(fp_cum) else 0.0
        fn, pres = class_total_fn[class_name], class_total_pres[class_name]

        class_accuracy[class_name] = tp_final / float(fn + pres) * 100 if (fn + pres) > 0 else 0.0
        class_far[class_name] = fp_final / float(fp_final + tp_final + fn) * 100 if (fp_final + tp_final + fn) > 0 else 0.0

    if total_iou_chunks:
        total_ious = np.concatenate(total_iou_chunks, axis=0) if total_pres > 0 else np.empty((0,), dtype=np.float32)
        total_tp = np.concatenate(total_tp_chunks, axis=0) if total_pres > 0 else np.empty((0,), dtype=np.float32)
        total_fp = np.concatenate(total_fp_chunks, axis=0) if total_pres > 0 else np.empty((0,), dtype=np.float32)
    else:
        total_ious = np.empty((0,), dtype=np.float32)
        total_tp = np.empty((0,), dtype=np.float32)
        total_fp = np.empty((0,), dtype=np.float32)

    sorted_idx = np.argsort(-total_ious) if total_ious.size else np.empty((0,), dtype=np.int64)
    all_tp = np.cumsum(total_tp[sorted_idx]) if sorted_idx.size else np.array([], dtype=np.float32)
    all_fp = np.cumsum(total_fp[sorted_idx]) if sorted_idx.size else np.array([], dtype=np.float32)

    recall = all_tp / float(total_gts) if total_gts > 0 else np.zeros(len(all_tp), dtype=np.float32)
    precision = all_tp / np.arange(1, len(all_tp) + 1, dtype=np.float32) if len(all_tp) else np.array([], dtype=np.float32)

    mAP = float(calculate_ap(precision, recall)) if len(precision) else 0.0
    accuracy = float(all_tp[-1] / float(total_fn + total_pres)) if (total_fn + total_pres) > 0 and len(all_tp) else 0.0
    far = float(all_fp[-1] / float(all_fp[-1] + all_tp[-1] + total_fn)) if len(all_fp) and (all_fp[-1] + all_tp[-1] + total_fn) > 0 else 0.0
    f1_score = float(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1])) if len(precision) and (precision[-1] + recall[-1]) > 0 else 0.0

    return (float(precision[-1] * 100) if len(precision) else 0.0,
            float(recall[-1] * 100) if len(recall) else 0.0,
            float(mAP * 100),
            float(accuracy * 100),
            float(far * 100),
            class_precision, class_recall, class_AP, class_accuracy, class_far, gt_counter_per_class, float(f1_score * 100))
