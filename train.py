import os
import sys
import warnings

# Ensure workspace root is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import NPY_datasets
from lbunet import LBUNet
from engine import train_one_epoch, val_one_epoch, test_one_epoch
from utils import get_logger, log_config_info, set_seed, get_optimizer, get_scheduler
from config_setting import setting_config

warnings.filterwarnings("ignore")


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    logger = get_logger('train', log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=False)

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    if config.network == 'lbunet':
        model = LBUNet(num_classes=model_cfg['num_classes'], 
                        input_channels=model_cfg['input_channels'], 
                        c_list=model_cfg['c_list'], 
                        )
    else: 
        raise Exception('network is not right!')
    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.3f}M")
    logger.info(f"Model parameters: {total_params / 1e6:.3f}M")

    print('#----------Preparing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler('cuda') if getattr(config, 'amp', False) else None

    print('#----------Set other params----------#')
    best_score = 0.0
    best_dice = 0.0
    best_acc = 0.0
    best_miou = 0.0
    best_epoch = 1
    start_epoch = 1
    patience = getattr(config, 'patience', 10)
    patience_counter = 0

    step = 0
    print(f'#----------Training (Max Epochs: {config.epochs}, Early Stopping Patience: {patience})----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            scaler=scaler
        )

        f1_or_dsc, accuracy, total_miou = val_one_epoch(
            val_loader,
            model,
            config.criterion,
            epoch,
            logger,
            config
        )

        # Composite score prioritizing Dice/F1 and mIoU
        score = f1_or_dsc + total_miou

        if score > best_score:
            best_score = score
            best_dice = f1_or_dsc
            best_acc = accuracy
            best_miou = total_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            print(f"--> Saved new BEST checkpoint at epoch {epoch}: Dice={best_dice*100:.2f}%, Acc={best_acc*100:.2f}%, mIoU={best_miou*100:.2f}%")
            logger.info(f"--> Saved new BEST checkpoint at epoch {epoch}: Dice={best_dice*100:.2f}%, Acc={best_acc*100:.2f}%, mIoU={best_miou*100:.2f}%")
        else:
            patience_counter += 1
            print(f"--> No improvement for {patience_counter}/{patience} epoch(s). (Best: Epoch {best_epoch}, Dice={best_dice*100:.2f}%)")
            logger.info(f"--> No improvement for {patience_counter}/{patience} epoch(s). (Best: Epoch {best_epoch}, Dice={best_dice*100:.2f}%)")
            if patience_counter >= patience:
                print(f"\n#----------Early Stopping Triggered at Epoch {epoch} (Patience {patience} Reached)----------#")
                logger.info(f"#----------Early Stopping Triggered at Epoch {epoch} (Patience {patience} Reached)----------#")
                break

    print(f'\n#----------Training Complete (Best Epoch: {best_epoch})----------#')
    best_ckpt_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_ckpt_path):
        print(f'#----------Final Testing with Best Checkpoint (Epoch {best_epoch})----------#')
        best_weight = torch.load(best_ckpt_path, map_location=torch.device('cuda:0'))
        model.load_state_dict(best_weight)
        test_one_epoch(
            val_loader,
            model,
            config.criterion,
            logger,
            config,
            path='ultimate',
            use_tta=True
        )


if __name__ == '__main__':
    config = setting_config
    main(config)
