from pathlib import Path
from tqdm import tqdm

from logging import getLogger
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

logger = getLogger(__name__)

class SemanticSegmentation(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.network = kwargs['network']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.train_loader, self.test_loader = kwargs['data_loaders']
        self.metrics = kwargs['metrics']
        self.vis_img = kwargs['vis_img']
        self.img_size = kwargs['img_size']
        self.writer = kwargs['writer']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']
        self.img_outdir = kwargs['img_outdir']

    def train(self, n_epochs, start_epoch=0):

        best_test_iou = 0

        for epoch in range(start_epoch, n_epochs):
            logger.info(f'\n\n==================== Epoch: {epoch} ====================')
            logger.info('### train:')
            self.network.train()

            train_loss = 0
            pred_list = []
            target_list = []

            with tqdm(self.train_loader, ncols=100) as pbar:
                for idx, (inputs, targets, img_paths_) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # outputs = self.network(inputs)
                    outputs, output_auxs  = self.network(inputs)

                    loss = self.criterion((outputs, output_auxs), targets.long())

                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_loss += loss.item()

                    preds = torch.softmax(outputs, 1).max(1)[1]

                    ### metrics update
                    pred_list.append(preds.cpu().detach().clone())
                    target_list.append(targets.cpu().detach().clone())

                    ### logging train loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss/(idx+1))))

            if epoch % self.save_ckpt_interval == 0:
                logger.info('\nsaving checkpoint...')
                self._save_ckpt(epoch, train_loss/(idx+1))

            logger.info('\ncalculate metrics...')
            preds = torch.cat([p for p in pred_list], axis=0)
            targets = torch.cat([t for t in target_list], axis=0)
            self.metrics.calc_metrics(preds, targets, train_loss/(idx+1), epoch, mode='train')
            self.metrics.initialize()

            ### test
            logger.info('\n### test:')
            test_mean_iou = self.test(epoch)

            if test_mean_iou > best_test_iou:
                logger.info(f'\nsaving best checkpoint (epoch: {epoch})...')
                best_test_iou = test_mean_iou
                self._save_ckpt(epoch, train_loss/(idx+1), mode='best')


    def test(self, epoch, inference=False):
        self.network.eval()

        test_loss = 0
        img_path_list = []
        pred_list = []
        target_list = []

        with torch.no_grad():
            with tqdm(self.test_loader, ncols=100) as pbar:
                for idx, (inputs, targets, img_paths) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # outputs = self.network(inputs)
                    outputs, output_auxs  = self.network(inputs)

                    loss = self.criterion((outputs, output_auxs), targets.long())

                    self.optimizer.zero_grad()

                    test_loss += loss.item()

                    img_path_list.extend(img_paths)

                    preds = torch.softmax(outputs, 1).max(1)[1]

                    ### metrics update
                    pred_list.append(preds.cpu().detach().clone())
                    target_list.append(targets.cpu().detach().clone())

                    ### logging test loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(test_loss/(idx+1))))

            ### metrics
            logger.info('\ncalculate metrics...')
            preds = torch.cat([p for p in pred_list], axis=0)
            targets = torch.cat([t for t in target_list], axis=0)
            self.metrics.calc_metrics(preds, targets, test_loss/(idx+1), epoch, mode='test')
            test_mean_iou = self.metrics.mean_iou

            ### show images on tensorboard
            self._show_imgs(img_path_list[:2], targets[:2], preds[:2], epoch, prefix='val')

            ### save result images
            if inference:
                logger.info('\nsaving images...')
                self._save_images(img_path_list, preds)

            self.metrics.initialize()

        return test_mean_iou

    def _save_ckpt(self, epoch, loss, mode=None, zfill=4):
        if isinstance(self.network, nn.DataParallel):
            network = self.network.module
        else:
            network = self.network

        if mode == 'best':
            ckpt_path = self.ckpt_dir / 'best_iou_ckpt.pth'
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'network': network,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, ckpt_path)

    def _show_imgs(self, img_paths, targets, preds, epoch, prefix='train'):
        """Show result image on Tensorboard

        Parameters
        ----------
        img_paths : list
            original image path
        preds : tensor
            [1, img_size, img_size] ([mini-batch, height, width])
        prefix : str, optional
            'train' or 'test', by default 'train'
        """

        for i, img_path in enumerate(img_paths):
            target = targets[i]
            pred = preds[i]

            orig_img = Image.open(img_path).resize((pred.shape[0], pred.shape[1]))
            orig_img = transforms.functional.to_tensor(orig_img)

            target_img = self.vis_img.decode_segmap(target)
            target_img = transforms.functional.to_tensor(target_img)

            pred_img = self.vis_img.decode_segmap(pred)
            pred_img = transforms.functional.to_tensor(pred_img)
            
            self.writer.add_image(f'{prefix}/original_{i}', orig_img, epoch)
            self.writer.add_image(f'{prefix}/target_{i}', target_img, epoch)
            self.writer.add_image(f'{prefix}/result_{i}', pred_img, epoch)

    def _save_images(self, img_paths, preds):
        """Save Image

        Parameters
        ----------
        img_paths : list
            original image paths
        preds : tensor
            [1, 21, img_size, img_size] ([mini-batch, n_classes, height, width])
        """

        for i, img_path in enumerate(img_paths):
            # preds[i] has background label 0, so exclude background class
            pred = preds[i]

            annotated_img = self.vis_img.decode_segmap(pred)

            width = Image.open(img_paths).size[0]
            height = Image.open(img_paths).size[1]

            annotated_img = annotated_img.resize((width, height), Image.NEAREST)

            outpath = self.img_outdir / Path(img_path).name
            self.vis_img.save_img(annotated_img, outpath)