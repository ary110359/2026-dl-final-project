import copy
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

from utils_mask import update_historical_mask, partition_regions
from generate_mask import generate_mask_for_class

def main():
    args = arg_parser.parse_args()

    device = utils.get_device(args)
    args.device = device

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    args.class_to_replace = None  # 改

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
    
    criterion = nn.CrossEntropyLoss() # 改

    # =========================================================================
    # 🚀 [新增] 準備連續遺忘 (Sequential Unlearning) 的環境變數
    # 如果你們有在 arg_parser 裡加 args.forget_sequence，可以改寫成從 args 讀取
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
        # ✂️ [修改] 第 1 步：動態切分 Dataset (取代原本龐大又冗長的 try-except 區塊)
        # =========================================================================
        forget_dataset = copy.deepcopy(original_dataset)
        retain_dataset = copy.deepcopy(original_dataset)
        
        # 統一轉換為 NumPy 陣列方便做 Boolean Indexing
        targets = np.array(original_dataset.targets) if hasattr(original_dataset, 'targets') else np.array(original_dataset.labels)
        
        # 1. 切出 Forget Dataset (只保留當前要忘記的類別)
        mask_forget = (targets == current_class)
        if hasattr(forget_dataset, 'data'):
            forget_dataset.data = forget_dataset.data[mask_forget]
        else: # 對應 TinyImageNet 的 .imgs
            forget_dataset.imgs = np.array(forget_dataset.imgs)[mask_forget].tolist()
            
        if hasattr(forget_dataset, 'targets'):
            forget_dataset.targets = targets[mask_forget].tolist()
        else:
            forget_dataset.labels = targets[mask_forget].tolist()

        # 2. 切出 Retain Dataset (保留集：排除當前類別，且「絕對不能」包含以前忘過的)
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

        # 把切好的 Dataset 裝進卡車 (DataLoader) 裡
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        
        print(f"📦 準備完畢：Retain 資料量 {len(retain_dataset)}, Forget 資料量 {len(forget_dataset)}")
        
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
        )

        # =========================================================================
        # 🧠 [修改] 第 2 步：載入模型並計算動態 Mask 與四大區域
        # =========================================================================
        
        # 第一輪如果是 Resume，就載入起點模型；之後的每一輪，模型都是繼承上一輪更新過的狀態
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
        # 這裡的 mask_ratio 可以改成從 args 抓，預設為 0.05
        mask_ratio = args.mask_ratio if hasattr(args, 'mask_ratio') else 0.05 
        current_mask = generate_mask_for_class(model, forget_loader, criterion, mask_ratio, args)

        print("🧩 正在與歷史遮罩比對，進行 Region Partition...")
        regions = partition_regions(current_mask, historical_mask)

        # =========================================================================
        # ⚙️ [修改] 第 3 步：執行真正的 Unlearn 手術
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
        # 💾 [修改] 第 4 步：更新歷史記憶，並儲存本輪模型與評估結果
        # =========================================================================
        forgotten_classes.append(current_class)
        historical_mask = update_historical_mask(historical_mask, current_mask)
        
        print(f"📊 正在評估第 {round_idx+1} 輪 (遺忘類別 {current_class}) 的表現...")
        evaluation_result = {}
        accuracy = {}
        
        # 跑驗證集算出各個 Accuracy
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"   => {name} acc: {val_acc}")
        
        evaluation_result["accuracy"] = accuracy
        
        # (選擇性) 跑 MIA 防禦評估 (這裡直接保留作者原版的 code)
        test_len = len(test_loader.dataset)
        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=args.batch_size, shuffle=False)

        utils.dataset_convert_to_test(test_loader, args)
        utils.dataset_convert_to_test(forget_loader, args)

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        
        # 每一輪結束時存檔，避免中斷
        # 你們可以修改 args.save_dir，讓它存成 "checkpoint_round_1.pth" 這種格式
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
        print(f"✅ 第 {round_idx+1} 輪結束！")


if __name__ == "__main__":
    main()
