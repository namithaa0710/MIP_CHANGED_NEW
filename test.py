import os
import sys
import glob
import warnings

# Ensure workspace root is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from torch.utils.data import DataLoader

from dataset import NPY_datasets
from lbunet import LBUNet
from engine import test_one_epoch
from utils import get_logger, log_config_info, set_seed
from config_setting import setting_config

warnings.filterwarnings("ignore")

def main(config):
    # Sort checkpoints by modification time (most recent first)
    checkpoints = glob.glob('results/**/best.pth', recursive=True)
    if not checkpoints:
        print("No checkpoint found in results/ directory.")
        return
    
    checkpoints.sort(key=os.path.getmtime)
    input_path = checkpoints[-1]
    work_dir = os.path.dirname(os.path.dirname(input_path)) + '/'
    config.work_dir = work_dir
    log_dir = os.path.join(work_dir, 'log')
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=0,
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

    if os.path.exists(input_path):
        print(f'#----------Testing with checkpoint: {input_path}----------#')
        best_weight = torch.load(input_path, map_location=torch.device('cuda:0'))
        model.load_state_dict(best_weight)
        
        # Test with Test-Time Augmentation
        print("\n--- Evaluation with Test-Time Augmentation (TTA) ---")
        test_one_epoch(
            val_loader,
            model,
            config.criterion,
            logger,
            config,
            path='ultimate',
            use_tta=True
        )
        
        output_dir = os.path.join(config.work_dir, 'outputs', config.datasets)
        print(f"\nOutputs and visualizations saved to: {output_dir}")


if __name__ == '__main__':
    config = setting_config
    main(config)
