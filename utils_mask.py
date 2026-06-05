"""本專案連續遺忘使用的 mask memory 工具。

[PROJECT MOD] 這個檔案不存在於原版 SalUn Classification。它負責實作
命中次數矩陣（Hit-Count Matrix），以及依命中次數衰減學習率需要用到的
參數區域切分。
"""

import torch

def update_historical_mask(historical_mask, current_mask):
    """
    [PROJECT MOD] 將原本 binary history mask 的概念升級成 hit count。

    功能與目的：
    不只記錄某個參數「是否曾經被選中」，而是累積它在連續遺忘過程中
    被 mask 選中的次數。這個次數之後會成為 alpha_conflict ** n 裡的 n。
    """
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
    """
    [PROJECT MOD] 將每個參數 tensor 切成連續遺忘需要的四種區域。

    功能與目的：
    原版 SalUn 只針對一次遺忘請求使用單一 saliency mask。本專案在連續
    遺忘中會比較當前 mask 和歷史 hit count：
    - new：現在被選中、以前沒被選中，允許完整更新。
    - conflict：現在和以前都被選中，使用 frequency-based decay。
    - old：以前被選中、現在沒被選中，凍結以保護舊遺忘結果。
    - other：現在和以前都沒被選中，在本方法中不更新。
    """
    regions = {}

    # [PROJECT MOD] 第一輪還沒有歷史衝突；當前 mask 內是新區域，
    # 其他位置則是其他區域。
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

    # [PROJECT MOD] 第二輪開始，透過當前 mask 和歷史命中次數記憶
    # 做集合操作。
    for k in current_mask.keys():
        curr = current_mask[k].bool()
        hist_count = historical_mask[k].to(current_mask[k].device).float()
        hist_seen = hist_count > 0
        updated_count = hist_count + current_mask[k].float()

        # [PROJECT MOD] 四種 region 的核心集合邏輯。
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
