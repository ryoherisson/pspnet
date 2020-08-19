import argparse
import yaml
from pathlib import Path
from datetime import datetime

from logging import getLogger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from torchsummary import summary

from utils.path_process import Paths
from utils.setup_logger import setup_logger
from data_process.data_path_process import make_datapath_list
from data_process.dataloader import DataTransform, VOCDataset
from modeling.pspnet.pspnet import PSPNet
from modeling.criterion.psploss import PSPLoss

logger = getLogger(__name__)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', type=str, default='./configs/default.yml')
    parser.add_argument('--inference', action='store_true', default=False)
    args = parser.parse_args()
    return args

def main(args):

    with open(args.configfile) as f:
        configs = yaml.safe_load(f)

    ## path process (path definition, make directories)
    now = datetime.now().isoformat()
    log_dir = Path(configs['log_dir']) / now
    paths = Paths(log_dir=log_dir)

    ### setup logs and summary writer ###
    setup_logger(logfile=paths.logfile)

    writer = SummaryWriter(str(paths.summary_dir))

    ### setup GPU or CPU ###
    if configs['n_gpus'] > 0 and torch.cuda.is_available():
        logger.info('CUDA is available! using GPU...\n')
        device = torch.device('cuda')
    else:
        logger.info('using CPU...\n')
        device = torch.device('cpu')

    ### Dataset ###
    logger.info('preparing dataset...')
    data_root = configs['data_root']
    logger.info(f'==> dataset path: {data_root}\n')

    train_img_list, train_annot_list, test_img_list, test_annot_list = make_datapath_list(rootpath=data_root, train_data=configs['train_txt'], test_data=configs['test_txt'])
    
    train_transform = DataTransform(input_size=configs['input_size'], color_mean=configs['color_mean'], color_std=configs['color_std'], mode='train')
    test_transform = DataTransform(input_size=configs['input_size'], color_mean=configs['color_mean'], color_std=configs['color_std'], mode='test')

    train_dataset = VOCDataset(train_img_list, train_annot_list, transform=train_transform, label_color_map=configs['label_color_map'])
    test_dataset = VOCDataset(test_img_list, test_annot_list, transform=test_transform, label_color_map=configs['label_color_map'])

    ### DataLoader ###
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False)

    ### Network ###
    logger.info('preparing network...')

    network = PSPNet(n_classes=configs['n_classes'], img_size=configs['input_size'], img_size_8=configs['input_size_8'])
    network = network.to(device)
    criterion = PSPLoss(aux_weight=configs['aux_weight'])
    optimizer = optim.Adam(network.parameters(), lr=configs['lr'], weight_decay=configs['decay'])

    logger.info('model summary: ')
    summary(network, input_size=(configs['n_channels'], configs['input_size'], configs['input_size']))

    ### Visualize Results ###

    ### Train or Inference ###


if __name__ == "__main__":
    args = parser()
    main(args)