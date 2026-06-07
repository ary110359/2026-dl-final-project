import copy
import csv
import os
from collections import OrderedDict

import arg_parser
import evaluation
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate

# [PROJECT MOD] 連續遺忘用的輔助工具。
# 這些 import 不在原版 SalUn 的 main_forget.py 中；用途是讓每一輪都能
# 產生新的 saliency mask、和歷史 hit count 比對，並把分區更新規則傳給 RL。
from utils_mask import update_historical_mask, partition_regions
from generate_mask import generate_mask_for_class
from utils import get_per_class_accuracy
from utils_mask import calculate_mask_saturation


def append_round_metrics_csv(
    csv_path,
    round_idx,
    current_class,
    forgotten_classes,
    per_class_acc,
    custom_metrics,
    mia_result,
):
    """[PROJECT MOD] 每一輪遺忘後追加一列評估資料，供畫圖程式讀取。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    row = {
        "step": round_idx + 1,
        "current_class": current_class,
        "forgotten_classes": "|".join(str(c) for c in forgotten_classes),
        "retain_acc": custom_metrics["retain_acc"],
        "current_forget_acc": custom_metrics["current_forget_acc"],
        "rebound_score": custom_metrics["rebound_score"],
        "max_rebound": custom_metrics["max_rebound"],
        "mask_saturation": custom_metrics["mask_saturation"],
        "mia_correctness": mia_result.get("correctness", 0.0) * 100,
        "mia_confidence": mia_result.get("confidence", 0.0) * 100,
        "mia_entropy": mia_result.get("entropy", 0.0) * 100,
        "mia_m_entropy": mia_result.get("m_entropy", 0.0) * 100,
        "mia_prob": mia_result.get("prob", 0.0) * 100,
    }
    for class_id in sorted(per_class_acc):
        row[f"class_{class_id}_accuracy"] = per_class_acc[class_id]

    mode = "w" if round_idx == 0 else "a"
    write_header = mode == "w" or not os.path.exists(csv_path)
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = arg_parser.parse_args()

    device = utils.get_device(args)
    args.device = device

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    # [PROJECT MOD] 關閉原版的一次性 class marking。
    # 本專案改由 args.forget_sequence 控制每一輪要遺忘的類別。
    args.class_to_replace = None

    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.to(device)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )
    
    criterion = nn.CrossEntropyLoss()

    # =========================================================================
    # [PROJECT MOD] 連續遺忘狀態。
    #
    # 原版 SalUn 是先產生一個 mask，再執行一次遺忘。本專案改成：
    # forget_sequence 定義要依序遺忘的多個 class id；
    # forgotten_classes 記錄已經不能再放進 retain set 的類別；
    # historical_mask 記錄每個參數跨輪被 mask 選中的累積次數。
    # =========================================================================
    forget_sequence = [int(x) for x in args.forget_sequence.split(',')] if hasattr(args, 'forget_sequence') else [0, 1, 2, 3, 4]
    forgotten_classes = []
    historical_mask = {}

    original_dataset = copy.deepcopy(train_loader_full.dataset)
    for round_idx, current_class in enumerate(forget_sequence):
        print("\n" + "="*60)
        print(f"🚀 開始第 {round_idx+1} 輪：準備遺忘類別 {current_class}")
        print(f"📚 目前已經忘記的類別：{forgotten_classes}")
        print("="*60)
        
        # =========================================================================
        # [PROJECT MOD] 第 1 步：依照本輪類別動態切分資料集。
        #
        # 功能與目的：
        # forget set 只包含當前要遺忘的類別；retain set 會排除當前類別和
        # 所有過去已遺忘類別，避免舊遺忘目標在後續輪次被重新強化。
        # =========================================================================
        forget_dataset = copy.deepcopy(original_dataset)
        retain_dataset = copy.deepcopy(original_dataset)
        
        # [PROJECT MOD] 統一 CIFAR/SVHN/TinyImageNet 的 label 欄位格式。
        targets = np.array(original_dataset.targets) if hasattr(original_dataset, 'targets') else np.array(original_dataset.labels)
        
        # [PROJECT MOD] Forget Dataset：只保留當前目標類別。
        mask_forget = (targets == current_class)
        if hasattr(forget_dataset, 'data'):
            forget_dataset.data = forget_dataset.data[mask_forget]
        else: # 對應 TinyImageNet 的 .imgs
            forget_dataset.imgs = np.array(forget_dataset.imgs)[mask_forget].tolist()
            
        if hasattr(forget_dataset, 'targets'):
            forget_dataset.targets = targets[mask_forget].tolist()
        else:
            forget_dataset.labels = targets[mask_forget].tolist()

        # [PROJECT MOD] Retain Dataset：排除當前類別與歷史遺忘類別。
        mask_retain = (targets != current_class)
        for f_cls in forgotten_classes:
            mask_retain = mask_retain & (targets != f_cls)
            
        if hasattr(retain_dataset, 'data'):
            retain_dataset.data = retain_dataset.data[mask_retain]
        else:
            retain_dataset.imgs = np.array(retain_dataset.imgs)[mask_retain].tolist()
            
        if hasattr(retain_dataset, 'targets'):
            retain_dataset.targets = targets[mask_retain].tolist()
        else:
            retain_dataset.labels = targets[mask_retain].tolist()

        # [PROJECT MOD] 依照本輪切分結果重建 DataLoader。
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        
        print(f"📦 準備完畢：Retain 資料量 {len(retain_dataset)}, Forget 資料量 {len(forget_dataset)}")
        
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
        )

        # =========================================================================
        # [PROJECT MOD] 第 2 步：產生本輪 mask，並切分參數區域。
        #
        # 功能與目的：
        # 原版 SalUn 使用預先算好的單次 mask_path；本專案每一輪都根據
        # 當前 forget_loader 計算 M_t，再和歷史 hit-count matrix 比對，
        # 分出 new / conflict / old / other 四種區域。
        # =========================================================================
        
        # [PROJECT MOD] 起始 checkpoint 只在第一輪載入；後續輪次沿用前一輪
        # 已經遺忘後的模型狀態。
        if round_idx == 0:
            if args.resume:
                checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
                if checkpoint is not None:
                    model, evaluation_result = checkpoint
            else:
                checkpoint = torch.load(args.model_path, map_location=device)
                if "state_dict" in checkpoint.keys():
                    checkpoint = checkpoint["state_dict"]
                if args.unlearn != "retrain":
                    model.load_state_dict(checkpoint, strict=False)

        print("🔍 正在計算當前類別的 Saliency Mask (M_t)...")
        # [PROJECT MOD] mask_ratio 控制 M_t 的稀疏程度；數值越小越能降低
        # 長序列遺忘中的 mask saturation 風險。
        mask_ratio = args.mask_ratio if hasattr(args, 'mask_ratio') else 0.05 
        current_mask = generate_mask_for_class(model, forget_loader, criterion, mask_ratio, args)

        print("🧩 正在與歷史遮罩比對，進行 Region Partition...")
        regions = partition_regions(current_mask, historical_mask)

        # =========================================================================
        # [PROJECT MOD] 第 3 步：執行遺忘，並在 RL 中啟用區域控制。
        #
        # 功能與目的：
        # RL 會接收 regions 和 alpha_conflict，進行參數級 post-update
        # correction；其他 unlearning 方法則維持原版呼叫方式。
        # =========================================================================
        print(f"⚡ 開始執行 Unlearn 演算法: {args.unlearn}")
        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        
        if args.unlearn == "RL":
            unlearn_method(
                unlearn_data_loaders,
                model,
                criterion,
                args,
                regions=regions,
                alpha_conflict=args.alpha_conflict,
            )
        else:
            unlearn_method(unlearn_data_loaders, model, criterion, args)
        
        # =========================================================================
        # [PROJECT MOD] 第 4 步：更新 hit-count 記憶，並評估本輪結果。
        # =========================================================================
        forgotten_classes.append(current_class)
        historical_mask = update_historical_mask(historical_mask, current_mask)
        
        print(f"\n📊 正在評估第 {round_idx+1} 輪 (已遺忘類別: {forgotten_classes}) 的表現...")
        evaluation_result = {}
        
        # 2. 跑一次 Test Loader，拿到 0~9 所有的準確率成績單
        utils.dataset_convert_to_test(test_loader.dataset, args)
        per_class_acc = get_per_class_accuracy(model, test_loader, args.num_classes, device)
        
        # 3. 計算自訂四大核心指標
        current_forget_acc = per_class_acc[current_class]
        
        retain_classes = [c for c in range(args.num_classes) if c not in forgotten_classes]
        retain_acc = sum([per_class_acc[c] for c in retain_classes]) / len(retain_classes) if retain_classes else 0.0
        
        rebound_score, max_rebound = 0.0, 0.0
        old_forgotten = forgotten_classes[:-1]
        if len(old_forgotten) > 0:
            old_accs = [per_class_acc[c] for c in old_forgotten]
            rebound_score = sum(old_accs) / len(old_accs)
            max_rebound = max(old_accs)
            
        mask_saturation = calculate_mask_saturation(historical_mask)

        # ---------------------------------------------------------
        # 4. 執行 SVC_MIA 攻擊測試 (原作者邏輯)
        # ---------------------------------------------------------
        test_len = len(test_loader.dataset)
        # 避免 retain_dataset 數量比 test_len 少而報錯
        shadow_train_size = min(test_len, len(retain_dataset)) 
        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(shadow_train_size)))
        shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=args.batch_size, shuffle=False)

        utils.dataset_convert_to_test(test_loader.dataset, args)
        utils.dataset_convert_to_test(forget_loader.dataset, args)

        mia_result = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        
        # 抓取 confidence 和 prob 的攻擊準確率
        mia_conf = mia_result.get('confidence', 0.0) * 100
        mia_prob = mia_result.get('prob', 0.0) * 100

        # ---------------------------------------------------------
        # 5. 印出漂漂亮亮的報告 (包含 MIA)
        # ---------------------------------------------------------
        print(f"    [Retain Accuracy]:         {retain_acc:.2f}% ")
        print(f"    [Current Forget Accuracy]: {current_forget_acc:.2f}% ")
        if len(old_forgotten) > 0:
            print(f"    [Rebound Score]:           {rebound_score:.2f}% (Max: {max_rebound:.2f}%)")
        print(f"    [Mask Saturation Ratio]:   {mask_saturation*100:.2f}%")
        print(f"     [MIA Attack (Confidence)]:  {mia_conf:.2f}% ")
        print(f"     [MIA Attack (Prob)]:        {mia_prob:.2f}% ")

        # 6. 存入 evaluation_result
        custom_metrics = {
            "retain_acc": retain_acc,
            "current_forget_acc": current_forget_acc,
            "rebound_score": rebound_score,
            "max_rebound": max_rebound,
            "mask_saturation": mask_saturation
        }
        evaluation_result["SVC_MIA_forget_efficacy"] = mia_result
        evaluation_result["custom_metrics"] = custom_metrics
        evaluation_result["per_class_accuracy"] = per_class_acc

        metrics_csv = os.path.join(args.save_dir, "sequential_metrics.csv")
        append_round_metrics_csv(
            metrics_csv,
            round_idx,
            current_class,
            forgotten_classes,
            per_class_acc,
            custom_metrics,
            mia_result,
        )
        print(f"     [CSV Metrics]:             {metrics_csv}")
        
        # 每一輪結束時存檔，避免中斷
        # 你們可以修改 args.save_dir，讓它存成 "checkpoint_round_1.pth" 這種格式
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
        print(f" 第 {round_idx+1} 輪結束")


if __name__ == "__main__":
    main()
