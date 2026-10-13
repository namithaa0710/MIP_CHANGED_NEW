import os
import sys

# Ensure workspace root is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from torchvision import transforms
from utils import (
    GT_BceDiceLoss,
    myNormalize,
    myToTensor,
    myRandomHorizontalFlip,
    myRandomVerticalFlip,
    myRandomRotation,
    myResize
)

from datetime import datetime

class setting_config:

    network = 'lbunet'
    model_config = {
        'num_classes': 1, 
        'input_channels': 3, 
        'c_list': [16, 24, 32, 48, 64, 96], 
        'bridge': True,
        'gt_ds': True,
    }

    datasets = 'isic18' 
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    elif datasets == 'BUSI':
        data_path = './data/Dataset_BUSI_with_GT/'
    else:
        raise Exception('datasets in not right!')

    criterion = GT_BceDiceLoss(wb=1, wd=1)
    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 2
    seed = 42
    world_size = None
    rank = None
    amp = True
    gpu_id = '0'
    batch_size = 8
    epochs = 30
    patience = 10

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 1
    save_interval = 20
    threshold = 0.5

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'AdamW':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-4
        amsgrad = False
    elif opt == 'Adam':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-4
        amsgrad = False
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 1e-4
        dampening = 0
        nesterov = True
    else:
        lr = 0.001
        weight_decay = 1e-4
    
    sch = 'WP_CosineLR'
    if sch == 'WP_CosineLR':
        warm_up_epochs = 5
    elif sch == 'CosineAnnealingLR':
        T_max = epochs
        eta_min = 1e-5
        last_epoch = -1
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 20
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 5
        gamma = 0.1
        milestones = [35, 50]
    elif sch == 'StepLR':
        step_size = epochs // 3
        gamma = 0.5
        last_epoch = -1
