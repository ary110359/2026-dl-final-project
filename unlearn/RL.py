import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .impl import iterative_unlearn

@torch.no_grad()
def post_update_correction(model, theta_before, regions, alpha):
    """
    強迫參數更新幅度遵循：
    New Region: 1.0 倍
    Conflict Region: alpha^n 倍 (n = cumulative hit count)
    Old Region: 0.0 倍 (鎖死)
    Other Region: 0.0 倍 (鎖死)
    """
    for name, param in model.named_parameters():
        # 如果這個參數不需要梯度，或者不在我們的遮罩管轄範圍內，就跳過
        if param.grad is None or name not in regions:
            continue
            
        before = theta_before[name]
        after = param.data
        delta = after - before  # Optimizer 原本想走的步伐
        region_masks = regions[name]
        
        # 把遮罩轉成跟參數一樣的 Device 和 Type
        r_new = region_masks['new'].to(device=param.device, dtype=param.dtype)
        r_conflict = region_masks['conflict'].to(device=param.device, dtype=param.dtype)
        r_old = region_masks['old'].to(device=param.device, dtype=param.dtype)
        r_other = region_masks['other'].to(device=param.device, dtype=param.dtype)
        hit_count = region_masks['hit_count'].to(device=param.device, dtype=param.dtype)
        conflict_scale = torch.pow(torch.full_like(hit_count, alpha), hit_count)
        
        # 建立倍率矩陣
        scale = (r_new * 1.0) + (r_conflict * conflict_scale) + (r_old * 0.0) + (r_other * 0.0)
                
        # 強制修正：退回原點，只走 (期望的倍率 * 步伐)
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
    
    # 這是為了相容舊版 baseline (如果沒有傳入 regions，就沿用原作者的方法)
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
            # 注意：這裡原本作者有個 bug，迴圈還沒開始就用了 i，我把它改成直接抓 current_step
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
            # ⚠️ [修改] 更新參數與防呆機制
            # ==========================================
            if regions:
                # [Ours] 在更新前，備份目前的參數
                with torch.no_grad():
                    theta_before = {name: param.data.clone() for name, param in model.named_parameters() if name in regions}
            elif mask:
                # [Baseline] 套用舊的梯度遮罩
                _apply_mask_to_grads(model, mask)
            
            optimizer.step()
            
            if regions:
                # [Ours] 在更新後，把亂跑的參數拉回我們規定的比例
                post_update_correction(model, theta_before, regions, alpha=alpha_conflict)
            elif mask:
                # [Baseline] 舊的還原機制
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
        
        # 1. 洗腦 Forget Loader
        for i, (image, target) in enumerate(forget_loader):
            image = image.to(device)
            target = torch.randint(0, args.num_classes, target.shape, device=device)
            
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            # ⚠️ [修改] 套用我們的參數修正機制
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
            
        # 2. 護腦 Retain Loader
        for i, (image, target) in enumerate(retain_loader):
            image = image.to(device)
            target = target.to(device)
            
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            # ⚠️ [修改] 套用我們的參數修正機制
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
