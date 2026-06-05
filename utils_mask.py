import torch

def update_historical_mask(historical_mask, current_mask):
    """Accumulate how many times each parameter has appeared in a mask."""
    current_counts = {k: v.float().clone() for k, v in current_mask.items()}
    if not historical_mask:
        return current_counts

    new_hist = {}
    for k in current_mask.keys():
        previous = historical_mask.get(k)
        if previous is None:
            new_hist[k] = current_counts[k]
        else:
            new_hist[k] = previous.to(current_counts[k].device).float() + current_counts[k]
    return new_hist

def partition_regions(current_mask, historical_mask):
    """Partition parameters and attach hit counts for frequency-based decay."""
    regions = {}

    # 如果是第一輪 (沒有歷史)，全部都是 New Region 和 Other Region
    if not historical_mask:
        for k in current_mask.keys():
            curr = current_mask[k].bool()
            zeros = torch.zeros_like(current_mask[k], dtype=torch.float32)
            regions[k] = {
                'new': curr.float(),
                'conflict': zeros.clone(),
                'old': zeros.clone(),
                'other': (~curr).float(),
                'hit_count': current_mask[k].float(),
            }
        return regions

    # 第二輪開始，嚴格劃分
    for k in current_mask.keys():
        curr = current_mask[k].bool()
        hist_count = historical_mask[k].to(current_mask[k].device).float()
        hist_seen = hist_count > 0
        updated_count = hist_count + current_mask[k].float()

        # 核心邏輯：集合操作
        new_m = curr & (~hist_seen)       # 現在要動，以前沒動過
        conflict_m = curr & hist_seen     # 現在要動，以前也動過
        old_m = (~curr) & hist_seen       # 現在不動，以前動過
        other_m = (~curr) & (~hist_seen)  # 現在不動，以前也沒動過

        regions[k] = {
            'new': new_m.float(),
            'conflict': conflict_m.float(),
            'old': old_m.float(),
            'other': other_m.float(),
            'hit_count': updated_count,
        }
    return regions
