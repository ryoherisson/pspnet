from tqdm import tqdm

from logging import getLogger
from collections import OrderedDict

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
                    self.metrics.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=train_loss)

                    ### logging train loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss)))

            if epoch % self.save_ckpt_interval == 0:
                logger.info('\nsaving checkpoint...')
                self._save_ckpt(epoch, train_loss/(idx+1))

            logger.info('\ncalculate metrics...')
            self.metrics.calc_metrics(epoch, mode='train')

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
                    self.metrics.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=test_loss)

                    ### logging test loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(test_loss)))

            ### metrics
            logger.info('\ncalculate metrics...')
            self.metrics.calc_metrics(epoch, mode='test') 
            test_mean_iou = self.metrics.mean_iou
            preds = self.metrics.preds

            ### show images on tensorboard
            # self._show_imgs(img_path_list[:2], preds[:2], self.img_size, epoch, prefix='val')

            ### save result images
            if inference:
                logger.info('\saving images...')
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

    def _show_imgs(self, img_paths, preds, img_size, epoch, prefix='train'):
        """Show result image on Tensorboard

        Parameters
        ----------
        img_paths : list
            original image path
        preds : tensor
            [1, 21, img_size, img_size] ([mini-batch, n_classes, height, width])
        img_size : int
            show img size
        prefix : str, optional
            'train' or 'test', by default 'train'
        """

        for i, img_path in enumerate(img_paths):
            pred = preds[i][1:]
            annotated_img = self.vis_img.decode_segmap(img_path, pred, img_size, img_size)
            annotated_img = transforms.functional.to_tensor(annotated_img)
            self.writer.add_image(f'{prefix}/results_{i}', annotated_img, epoch)

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

            outpath = self.img_outdir / img_path.name
            self.vis_img.save_img(annotated_img, outpath)