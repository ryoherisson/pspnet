"""PSPLoss
Reference
    Original author: YutaroOgawa
    https://github.com/YutaroOgawa/pytorch_advanced/blob/master/3_semantic_segmentation/3-7_PSPNet_training.ipynb
"""

import torch.nn as nn
import torch.nn.functional as F


class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):

        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[0], targets, reduction='mean')

        return loss + self.aux_weight * loss_aux