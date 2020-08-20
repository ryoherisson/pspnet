"""Metrics
calsulate ious and mean iou
"""

import csv
from logging import getLogger

import numpy as np
import pandas as pd

import torch


logger = getLogger(__name__)
pd.set_option('display.unicode.east_asian_width', True)

class Metrics(object):
    def __init__(self, **cfg):
        self.n_classes = cfg['n_classes']
        self.classes = ['background'] + cfg['classes'] # 0 is background
        self.img_size = cfg['img_size']
        self.writer = cfg['writer']
        self.metrics_dir = cfg['metrics_dir']

        self.initialize()

    def initialize(self):
        self.cmx = np.zeros((self.n_classes, self.n_classes))

    def calc_metrics(self, preds, targets, loss, epoch, mode):

        preds = preds.view(-1)
        targets = targets.view(-1)

        preds = preds.numpy()
        targets = targets.numpy()

        # calc histgram and make confusion matrix
        cmx = np.bincount(self.n_classes * targets.astype(int) 
                         + preds, minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        
        with np.errstate(invalid='ignore'):
            self.ious = np.diag(cmx) / (cmx.sum(axis=1) + cmx.sum(axis=0) - np.diag(cmx))
        
        self.loss = loss
        self.mean_iou = np.nanmean(self.ious)

        self.logging(epoch, mode)
        self.save_csv(epoch, mode)

    def logging(self, epoch, mode):
        logger.info(f'{mode} metrics...')
        logger.info(f'loss:         {self.loss}')

        # ious per class
        df = pd.DataFrame(index=self.classes)
        df['IoU'] = self.ious.tolist()
        logger.info(f'\nmetrics value per classes: \n{df}\n')

        # micro mean iou
        logger.info(f'mean iou:    {self.mean_iou}')

        # Change mode from 'test' to 'val' to change the display order from left to right to train and test.
        mode = 'val' if mode == 'test' else mode

        self.writer.add_scalar(f'loss/{mode}', self.loss, epoch)
        self.writer.add_scalar(f'mean_iou/{mode}', self.mean_iou, epoch)

    def save_csv(self, epoch, mode):
        csv_path = self.metrics_dir / f'{mode}_metrics.csv'

        if not csv_path.exists():
            with open(csv_path, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', f'{mode} loss', f'{mode} iou'])

        with open(csv_path, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, self.loss, self.mean_iou.item()])