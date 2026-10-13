import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from utils import save_imgs, compute_metrics


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer, scaler=None):
    model.train() 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()
        images, targets, points = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()
        points = points.cuda(non_blocking=True).float()

        if getattr(config, 'amp', False) and scaler is not None:
            with torch.amp.autocast('cuda'):
                gt_pre, key_points, out = model(images)
                loss = criterion(gt_pre, key_points, out, targets, points)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            gt_pre, key_points, out = model(images)
            loss = criterion(gt_pre, key_points, out, targets, points)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if writer is not None:
            writer.add_scalar('loss', loss.item(), global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}/{config.epochs}, iter:{iter}/{len(train_loader)}, loss: {np.mean(loss_list):.4f}, lr: {now_lr:.6f}'
            print(log_info)
            logger.info(log_info)

    scheduler.step() 
    return step


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    loss_list = []
    total_miou = 0.0
    total = 0
    gt_list = []
    pred_list = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f'Val {epoch}', leave=False):
            if len(data) == 3:
                img, msk, pnt = data
                img = img.cuda(non_blocking=True).float()
                msk = msk.cuda(non_blocking=True).float()
                pnt = pnt.cuda(non_blocking=True).float()
                if getattr(config, 'amp', False):
                    with torch.amp.autocast('cuda'):
                        gt_pre, key_points, out = model(img)
                        loss = criterion(gt_pre, key_points, out, msk, pnt)
                else:
                    gt_pre, key_points, out = model(img)
                    loss = criterion(gt_pre, key_points, out, msk, pnt)
                loss_list.append(loss.item())
            else:
                img, msk = data
                img = img.cuda(non_blocking=True).float()
                msk = msk.cuda(non_blocking=True).float()
                if getattr(config, 'amp', False):
                    with torch.amp.autocast('cuda'):
                        gt_pre, key_points, out = model(img)
                else:
                    gt_pre, key_points, out = model(img)

            gts = msk.squeeze(1).cpu().numpy()
            preds = out.squeeze(1).cpu().numpy()
            gt_list.append(gts)
            pred_list.append(preds)

            preds_flat = np.array(preds).reshape(-1)
            gts_flat = np.array(gts).reshape(-1)

            y_pre = np.where(preds_flat >= config.threshold, 1, 0)
            y_true = np.where(gts_flat >= 0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)
            total_miou += miou
            total += 1
            
    total_miou = total_miou / total if total > 0 else 0.0
    pred_arr = np.concatenate(pred_list, axis=0).reshape(-1)
    gt_arr = np.concatenate(gt_list, axis=0).reshape(-1)

    accuracy, sensitivity, specificity, f1_or_dsc = compute_metrics(gt_arr, pred_arr, threshold=config.threshold)
    val_loss_avg = np.mean(loss_list) if len(loss_list) > 0 else 0.0

    log_info = (f'val epoch: {epoch}, loss: {val_loss_avg:.4f}, '
                f'acc: {accuracy*100:.2f}%, dice/f1: {f1_or_dsc*100:.2f}%, '
                f'miou: {total_miou*100:.2f}%, sens: {sensitivity*100:.2f}%, spec: {specificity*100:.2f}%')
    print(log_info)
    logger.info(log_info)

    # Return composite evaluation score (f1_or_dsc, accuracy, total_miou)
    return f1_or_dsc, accuracy, total_miou


def test_one_epoch(test_loader, model, criterion, logger, config, path='ultimate', test_data_name=None, use_tta=True):
    model.eval()
    gt_list = []
    pred_list = []
    total_miou = 0.0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc='Testing', leave=True)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            if use_tta:
                # Test-Time Augmentation: Average prediction of original, horizontal flip, vertical flip
                if getattr(config, 'amp', False):
                    with torch.amp.autocast('cuda'):
                        _, _, out1 = model(img)
                        _, _, out2 = model(torch.flip(img, dims=[3]))
                        out2 = torch.flip(out2, dims=[3])
                        _, _, out3 = model(torch.flip(img, dims=[2]))
                        out3 = torch.flip(out3, dims=[2])
                        gt_pre, key_points, _ = model(img)
                else:
                    _, _, out1 = model(img)
                    _, _, out2 = model(torch.flip(img, dims=[3]))
                    out2 = torch.flip(out2, dims=[3])
                    _, _, out3 = model(torch.flip(img, dims=[2]))
                    out3 = torch.flip(out3, dims=[2])
                    gt_pre, key_points, _ = model(img)
                out = (out1 + out2 + out3) / 3.0
            else:
                if getattr(config, 'amp', False):
                    with torch.amp.autocast('cuda'):
                        gt_pre, key_points, out = model(img)
                else:
                    gt_pre, key_points, out = model(img)

            msk_np = msk.squeeze(1).cpu().numpy()
            out_np = out.squeeze(1).cpu().numpy()

            gt_list.append(msk_np)
            pred_list.append(out_np)
            
            y_pre = np.where(out_np.reshape(-1) >= config.threshold, 1, 0)
            y_true = np.where(msk_np.reshape(-1) >= 0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)

            total_miou += miou
            total += 1

            if i % config.save_interval == 0:
                save_imgs(img, msk_np, out_np, key_points, gt_pre, i, 
                          os.path.join(config.work_dir, 'outputs', config.datasets), 
                          config.datasets, config.threshold, test_data_name=test_data_name)

    total_miou = total_miou / total if total > 0 else 0.0

    pred_arr = np.concatenate(pred_list, axis=0).reshape(-1)
    gt_arr = np.concatenate(gt_list, axis=0).reshape(-1)

    accuracy, sensitivity, specificity, f1_or_dsc = compute_metrics(gt_arr, pred_arr, threshold=config.threshold)
    
    log_info = (f'\n#====================== TEST EVALUATION RESULTS ======================#\n'
                f'Pixel Accuracy : {accuracy*100:.2f}% ({accuracy:.4f})\n'
                f'Dice / F1 Score: {f1_or_dsc*100:.2f}% ({f1_or_dsc:.4f})\n'
                f'mIoU (Jaccard) : {total_miou*100:.2f}% ({total_miou:.4f})\n'
                f'Sensitivity    : {sensitivity*100:.2f}% ({sensitivity:.4f})\n'
                f'Specificity    : {specificity*100:.2f}% ({specificity:.4f})\n'
                f'#=====================================================================#')
    print(log_info)
    logger.info(log_info)
    return accuracy, f1_or_dsc, total_miou, sensitivity, specificity
