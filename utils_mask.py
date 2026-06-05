# 📄 Classification/utils_mask.py
import torch

def update_historical_mask(historical_mask, current_mask):
    """將 current_mask 併入 historical_mask 中 (取聯集 OR)"""
    if not historical_mask: # 第一輪，歷史是空的
        return {k: v.clone() for k, v in current_mask.items()}
        
    new_hist = {}
    for k in current_mask.keys():
        # bitwise OR: 只要以前或現在是 1，結果就是 1
        new_hist[k] = (historical_mask[k].bool() | current_mask[k].bool()).float()
    return new_hist

def partition_regions(current_mask, historical_mask):
    """根據 M_t 和 H 劃分出四大區域"""
    regions = {}
    
    # 如果是第一輪 (沒有歷史)，全部都是 New Region 和 Other Region
    if not historical_mask:
        for k in current_mask.keys():
            curr = current_mask[k].bool()
            regions[k] = {
                'new': curr.float(),
                'conflict': torch.zeros_like(curr).float(),
                'old': torch.zeros_like(curr).float(),
                'other': (~curr).float()
            }
        return regions

    # 第二輪開始，嚴格劃分
    for k in current_mask.keys():
        curr = current_mask[k].bool()
        hist = historical_mask[k].bool()
        
        # 核心邏輯：集合操作
        new_m = curr & (~hist)       # 現在要動，以前沒動過
        conflict_m = curr & hist     # 現在要動，以前也動過
        old_m = (~curr) & hist       # 現在不動，以前動過
        other_m = (~curr) & (~hist)  # 現在不動，以前也沒動過
        
        regions[k] = {
            'new': new_m.float(),
            'conflict': conflict_m.float(),
            'old': old_m.float(),
            'other': other_m.float()
        }
    return regions