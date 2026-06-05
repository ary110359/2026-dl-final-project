import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .impl import iterative_unlearn

@torch.no_grad()
def post_update_correction(model, theta_before, regions, alpha):
    """
    [PROJECT MOD] 連續 SalUn 使用的參數級 post-update correction。

    功能與目的：
    先讓 Adam/SGD 算出正常 optimizer step，接著重寫最後的參數值，讓每個
    座標都遵守 region policy：
    - 新區域：1.0x 更新，全力遺忘當前目標特徵。
    - 衝突區域：alpha^n 更新，n 是累積命中次數。
    - 舊區域：0.0x 更新，保護先前遺忘結果。
    - 其他區域：0.0x 更新，避免無關參數漂移。
    """
    for name, param in model.named_parameters():
        # [PROJECT MOD] 跳過不在 region partition 管轄範圍內的 tensor。
        if param.grad is None or name not in regions:
            continue
            
        before = theta_before[name]
        after = param.data
        delta = after - before  # Optimizer's original proposed step.
        region_masks = regions[name]
        
        r_new = region_masks['new'].to(device=param.device, dtype=param.dtype)
        r_conflict = region_masks['conflict'].to(device=param.device, dtype=param.dtype)
        r_old = region_masks['old'].to(device=param.device, dtype=param.dtype)
        r_other = region_masks['other'].to(device=param.device, dtype=param.dtype)
        hit_count = region_masks['hit_count'].to(device=param.device, dtype=param.dtype)
        conflict_scale = torch.pow(torch.full_like(hit_count, alpha), hit_count)
        
        # [PROJECT MOD] 建立每個參數座標自己的更新倍率矩陣。
        scale = (r_new * 1.0) + (r_conflict * conflict_scale) + (r_old * 0.0) + (r_other * 0.0)
                
        # [PROJECT MOD] 回到 theta_before，只套用被允許的那一段更新步伐。
        param.data.copy_(before + scale * delta)


def _apply_mask_to_grads(model, mask):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad *= mask[name]


def _restore_masked_params(model, mask, theta0, optimizer):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue

            mask_tensor = mask[name].to(device=param.device, dtype=param.dtype)
            inv_mask_tensor = 1 - mask_tensor
            if torch.count_nonzero(inv_mask_tensor) == 0:
                continue

            # Keep masked-out weights exactly at initialization value (theta0).
            param.data.mul_(mask_tensor).add_(theta0[name].to(param.device) * inv_mask_tensor)

            # Prevent momentum from reintroducing updates on masked-out coordinates.
            state = optimizer.state.get(param, None)
            if state is not None and "momentum_buffer" in state:
                state["momentum_buffer"].mul_(mask_tensor)


@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None, regions=None, alpha_conflict=0.1):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    device = getattr(args, "device", next(model.parameters()).device)
    
    # [PROJECT MOD] 向原版相容：如果沒有傳入 regions，就維持原本
    # mask-based SalUn 的行為。
    theta0 = None
    if mask and not regions:
        with torch.no_grad():
            theta0 = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if name in mask
            }
    
    if args.dataset == "cifar100" or args.dataset == "TinyImagenet":
        try:
            forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
        except:
            forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))
    
        retain_dataset = retain_loader.dataset
        train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            # [PROJECT MOD] 避免在迴圈變數 i 尚未建立前使用它。
            utils.warmup_lr(epoch, 1, optimizer, one_epoch_step=loader_len, args=args)
      
        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.to(device)
            target = target.to(device)
            output_clean = model(image)

            loss = criterion(output_clean, target)
      
            optimizer.zero_grad()
            loss.backward()
            
            # ==========================================
            # [PROJECT MOD] 區域感知 optimizer correction。
            # 在 optimizer.step() 前備份 theta，更新後再依照 region partition
            # 對每個座標做縮放或凍結。
            # ==========================================
            if regions:
                # [PROJECT MOD] 在 optimizer 更新前保存 theta_t。
                with torch.no_grad():
                    theta_before = {name: param.data.clone() for name, param in model.named_parameters() if name in regions}
            elif mask:
                # [原版相容] 套用原本的 gradient mask 流程。
                _apply_mask_to_grads(model, mask)
            
            optimizer.step()
            
            if regions:
                # [PROJECT MOD] 用 region-specific scale 重寫 theta_{t+1}。
                post_update_correction(model, theta_before, regions, alpha=alpha_conflict)
            elif mask:
                # [原版相容] 還原被 mask 排除的參數。
                _restore_masked_params(model, mask, theta0, optimizer)
            # ==========================================
      
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
      
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
      
            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                          epoch, i, loader_len, end-start, loss=losses, top1=top1))
                start = time.time()
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, 1, optimizer, one_epoch_step=loader_len, args=args)
        
        # [PROJECT MOD] Forget phase：對目標資料做 random-label training。
        for i, (image, target) in enumerate(forget_loader):
            image = image.to(device)
            target = torch.randint(0, args.num_classes, target.shape, device=device)
            
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            # [PROJECT MOD] 在 forget phase 套用同一套 region-aware correction。
            if regions:
                with torch.no_grad():
                    theta_before = {name: param.data.clone() for name, param in model.named_parameters() if name in regions}
            elif mask:
                _apply_mask_to_grads(model, mask)
            
            optimizer.step()
            
            if regions:
                post_update_correction(model, theta_before, regions, alpha=alpha_conflict)
            elif mask:
                _restore_masked_params(model, mask, theta0, optimizer)
            
        # [PROJECT MOD] Retain phase：保護剩餘類別的辨識能力。
        for i, (image, target) in enumerate(retain_loader):
            image = image.to(device)
            target = target.to(device)
            
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            # [PROJECT MOD] 在 retain phase 套用同一套 region-aware correction。
            if regions:
                with torch.no_grad():
                    theta_before = {name: param.data.clone() for name, param in model.named_parameters() if name in regions}
            elif mask:
                _apply_mask_to_grads(model, mask)
            
            optimizer.step()
            
            if regions:
                post_update_correction(model, theta_before, regions, alpha=alpha_conflict)
            elif mask:
                _restore_masked_params(model, mask, theta0, optimizer)
            
            output = output_clean.float()
            loss = loss.float()
            prec1 = utils.accuracy(output.data, target)[0]
            
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            
            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Time {3:.2f}'.format(
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()

    return top1.avg
