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
from utils.vis_img import VisImage
from data_process.data_path_process import make_datapath_list
from data_process.dataloader import DataTransform, VOCDataset
from modeling.semseg import SemanticSegmentation
from modeling.pspnet.pspnet import PSPNet
from modeling.criterion.psploss import PSPLoss
from modeling.metrics.metrics import Metrics

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
    
    train_transform = DataTransform(img_size=configs['img_size'], color_mean=configs['color_mean'], color_std=configs['color_std'], mode='train')
    test_transform = DataTransform(img_size=configs['img_size'], color_mean=configs['color_mean'], color_std=configs['color_std'], mode='test')

    train_dataset = VOCDataset(train_img_list, train_annot_list, transform=train_transform, label_color_map=configs['label_color_map'])
    test_dataset = VOCDataset(test_img_list, test_annot_list, transform=test_transform, label_color_map=configs['label_color_map'])

    ### DataLoader ###
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False)

    ### Network ###
    logger.info('preparing network...')

    network = PSPNet(n_classes=configs['n_classes'], img_size=configs['img_size'], img_size_8=configs['input_size_8'])
    network = network.to(device)
    criterion = PSPLoss(aux_weight=configs['aux_weight'])
    optimizer = optim.Adam(network.parameters(), lr=configs['lr'], weight_decay=configs['decay'])

    if configs['resume']:
        # Load checkpoint
        logger.info('==> Resuming from checkpoint...\n')
        if not Path(configs['resume']).exists():
            logger.info('No checkpoint found !')
            raise ValueError('No checkpoint found !')

        network.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        loss = ckpt['loss']
    else:
        logger.info('==> Building model...\n')
        start_epoch = 0 


    logger.info('model summary: ')
    summary(network, input_size=(configs['n_channels'], configs['img_size'], configs['img_size']))

    if configs['n_gpus'] > 1:
        network = nn.DataParallel(network)
        
    ### Metrics ###
    metrics_cfg = {
        'n_classes': configs['n_classes'],
        'classes': configs['classes'],
        'img_size': configs['img_size'],
        'writer': writer,
        'metrics_dir': paths.metrics_dir,
    }

    metrics = Metrics(**metrics_cfg)

    ### Visualize Results ###
    vis_img = VisImage(n_classes=configs['n_classes'], label_color_map=configs['label_color_map'])

    ### Train or Inference ###
    kwargs = {
        'device': device,
        'network': network,
        'optimizer': optimizer,
        'criterion': criterion,
        'data_loaders': (train_loader, test_loader),
        'metrics': metrics,
        'vis_img': vis_img,
        'img_size': configs['img_size'],
        'writer': writer,
        'save_ckpt_interval': configs['save_ckpt_interval'],
        'ckpt_dir': paths.ckpt_dir,
        'img_outdir': paths.img_outdir,
    }

    semantic_segmentaion = SemanticSegmentation(**kwargs)

    if args.inference:
        if not configs['resume']:
            logger.info('No checkpoint found for inference!')
        logger.info('mode: inference\n')
        semantic_segmentaion.test(epoch=start_epoch, inference=True)
    else:
        logger.info('mode: train\n')
        semantic_segmentaion.train(n_epochs=configs['n_epochs'], start_epoch=start_epoch)

if __name__ == "__main__":
    args = parser()
    main(args)