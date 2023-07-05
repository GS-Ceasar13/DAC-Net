"""
GDiceLoss
GDiceLossV2
SSLoss
SoftDiceLoss
IouLoss
TverskyLoss
FocalTversky_loss
AsymLoss
Dice_and_CE_loss
PenaltyGDiceLoss
Dice_and_Topk_loss
ExpLog_Loss
"""
import numpy as np
import torch
from torch import nn
from torch import einsum
from torch.autograd import Variable
from losses_pytorch.ND_Crossentropy import CrossentropyND  as CrossEntropy
from losses_pytorch.ND_Crossentropy import TopKLoss  as  TopkLoss
from losses_pytorch.ND_Crossentropy import WeightedCrossEntropyLoss  as  Weight_CrossEntropy_Loss
import torch.nn.functional as F

def soft_max(x):
    rpt = [1 for i in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(dim=1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=1, keepdim=True).repeat(*rpt)


# def sum_tensor(inp, axes, keepdim=False):
#     # axes = np.unique(axes).astype(int)
#     if keepdim:
#         for ax in axes:
#             inp = inp.sum(dim=int(ax), keepdim=True)
#     else:
#         for ax in axes:
#             inp = inp.sum(dim=int(ax), keepdim=False)
#     return inp
def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def tp_tn_fp_fn(net_out, target, axes=None, mask=None, square=False):
    """
    net_out : (b, c, h, w)
    targrt : (b, 1, h, w) or (b, h, w) or one_hot encoding (b, c, h, w)
    """
    num_class = net_out.size()[1]
    if axes is None:
        axes = tuple(range(2, len(net_out.shape)))

    shp_x = net_out.shape
    shp_y = target.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            target = target.view((shp_y[0], 1, *shp_y[1:]))
        if all([i == j for i, j in zip(net_out.shape, target.shape)]):
            one_hot = target
        else:
            idx = target.long()
            one_hot = torch.zeros(shp_x)
            if net_out.device.type == "cuda":
                one_hot = one_hot.cuda(net_out.device.index)
            one_hot.scatter_(1, idx, 1)

    tp = net_out * one_hot
    tn = (1 - net_out) * (1 - one_hot)
    fp = net_out * (1 - one_hot)
    fn = (1 - net_out) * one_hot

    if mask != None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        tn = tn ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=True).view(-1, num_class)
    tn = sum_tensor(tn, axes, keepdim=True).view(-1, num_class)
    fp = sum_tensor(fp, axes, keepdim=True).view(-1, num_class)
    fn = sum_tensor(fn, axes, keepdim=True).view(-1, num_class)

    return tp, tn, fp, fn


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_out, target):
        net_out=F.softmax(net_out,dim=1)
        if self.apply_nonlin != None:
            net_out = self.apply_nonlin(net_out)

        shp_x = net_out.shape
        shp_y = target.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))
            if all([i == j for i, j in zip(net_out.shape, target.shape)]):
                one_hot = target
            else:
                idx = target.long()
                one_hot = torch.zeros(shp_x)
                if net_out.device.type == "cuda":
                    one_hot = one_hot.cuda(net_out.device.index)
                one_hot.scatter_(1, idx, 1)
                t=one_hot.permute(0,2,3,1)
                # print(t)

        w: torch.Tensor = 1 / (einsum('bcxy->bc', one_hot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum('bcxy, bcxy->bc', net_out, one_hot)
        union: torch.Tensor = w * (einsum('bcxy->bc', net_out) + einsum('bcxy->bc', one_hot))
        divided: torch.Tensor = 2 * (einsum('bc->b', intersection) + self.smooth) / (einsum('bc->b', union) + self.smooth)
        GDLoss = divided.mean()

        return 1 - GDLoss


class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        if self.apply_nonlin != None:
            net_out = self.apply_nonlin(net_out)

        shp_x = net_out.shape
        shp_y = target.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view(shp_y[0], 1, *shp_y[1:])
            if all([i == j for i, j in zip(shp_x, shp_y)]):
                one_hot = target
            else:
                idx = target.long()
                one_hot = torch.zeros(shp_x)
                if net_out.device.type == "cuda":
                    one_hot = one_hot.cuda(net_out.device.index)
                one_hot = one_hot.scatter_(1, idx, 1)

        input = torch.flatten(net_out)
        target = torch.flatten(one_hot).float()
        # target_sum = target.sum(dim=-1)
        target_sum = target.sum(dim=1)

        class_weight = Variable(1 / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)
        intersection = (input * target).sum(dim=-1) * class_weight
        intersection = intersection.sum()
        denomimator = ((input + target).sum(dim=-1) * class_weight).sum()
        divided = -2 * intersection / denomimator.clamp(min=self.smooth)

        return 1+ divided


class SSLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        Sensitivity-Specifity loss
        paper: http://www.rogertam.ca/Brosch_MICCAI_2015.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py#L392
        """
        super(SSLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1  # weight parameter in SS paper

    def forward(self, net_output, gt, loss_mask=None):
        net_output = F.softmax(net_output, dim=1)
        shp_x = net_output.shape
        shp_y = gt.shape
        # class_num = shp_x[1]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - net_output) ** 2
        specificity_part = sum_tensor(squared_error * y_onehot, axes) / (sum_tensor(y_onehot, axes) + self.smooth)
        sensitivity_part = sum_tensor(squared_error * bg_onehot, axes) / (sum_tensor(bg_onehot, axes) + self.smooth)

        ss = self.r * specificity_part + (1 - self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()

        return ss


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5,
                 batch_dice=False, do_bg=True, square=False, loss_mask=None):
        super(SoftDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.square = square
        self.loss_mask = loss_mask

    def forward(self, net_out, target):
        shp_x = net_out.shape

        if self.apply_nonlin != None:
            net_out = self.apply_nonlin(net_out)
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        tp, tn, fp, fn = tp_tn_fp_fn(net_out, target, axes, mask=self.loss_mask, square=self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if self.do_bg is not True:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        dc = dc.mean()
        return 1- dc


class IouLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5,
                 batch_dice=False, do_bg=True, square=False, loss_mask=None):
        super(IouLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.square = square
        self.loss_mask = loss_mask

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        shp_x = net_out.shape

        if self.apply_nonlin != None:
            net_out = self.apply_nonlin(net_out)
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        tp, tn, fp, fn = tp_tn_fp_fn(net_out, target, axes, self.loss_mask, self.square)

        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if self.do_bg is not True:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return -torch.log(iou)



class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5,
                 batch_dice=False, do_bg=True, square=False, loss_mask=None, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.square = square
        self.loss_mask = loss_mask
        self.alpha = alpha
        self.beta = beta

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        shp_x = net_out.shape

        if self.apply_nonlin != None:
            net_out = self.apply_nonlin(net_out)
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        tp, tn, fp, fn = tp_tn_fp_fn(net_out, target, axes, self.loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.do_bg is not True:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return tversky


class FocalTversky_loss(nn.Module):
    def __init__(self, gamma=0.75):
        super(FocalTversky_loss, self).__init__()

        self.tversky_kwargs = TverskyLoss()
        self.gamma = gamma

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        tversky_loss = 1- self.tversky_kwargs(net_out, target)
        focaltversky_loss = torch.pow(tversky_loss, self.gamma)
        return focaltversky_loss


class AsymLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5,
                 batch_dice=False, do_bg=True, square=False, loss_mask=None, beta=1.5):
        super(AsymLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.square = square
        self.loss_mask = loss_mask
        self.beta = beta

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        shp_x = net_out.shape

        if self.apply_nonlin is not None:
            net_out = self.apply_nonlin(net_out)
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        tp , tn, fp, fn = tp_tn_fp_fn(net_out, target, axes, self.loss_mask, self.square)

        weight = (self.beta ** 2) / (1 + self.beta ** 2)
        asym_loss = (tp + self.smooth) / (tp + weight*fn + (1 - weight)*fp + self.smooth)

        if self.do_bg is not True:
            if self.batch_dice:
                asym_loss = asym_loss[1:]
            else:
                asym_loss = asym_loss[:, 1:]
        asym_loss = asym_loss.mean()

        return 1- asym_loss


class Dice_and_CE_loss(nn.Module):
    def __init__(self, aggregate='sum'):
        super(Dice_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossEntropy()
        self.dice = SoftDiceLoss()

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        dice_loss = self.dice(net_out, target)
        ce_loss = self.ce(net_out, target)

        if self.aggregate == 'sum':
            loss = dice_loss + ce_loss
        else:
            raise NotImplementedError('nah son')

        return loss


class PenaltyGDiceLoss(nn.Module):
    def __init__(self, k=2.5):
        super(PenaltyGDiceLoss, self).__init__()

        self.k = k
        self.GDice = GDiceLoss()

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        GDice_loss = self.GDice(net_out, target)
        panalty_GDice_loss = GDice_loss / (1 + self.k * (1-GDice_loss))

        return panalty_GDice_loss

class Dice_and_Topk_loss(nn.Module):
    def __init__(self, agregate='sum'):
        super(Dice_and_Topk_loss, self).__init__()

        self.agregate = agregate
        self.dice = SoftDiceLoss()
        self.topk = TopkLoss()

    def forward(self, net_out, target):
        dice_loss = self.dice(net_out, target)
        topk_loss = self.topk(net_out, target)
        if self.agregate == 'sum':
            dice_and_topk_loss = dice_loss + topk_loss
        else:
            raise NotImplementedError('nah son')
        return dice_and_topk_loss


class ExpLog_Loss(nn.Module):
    def __init__(self, gamma=0.3):
        super(ExpLog_Loss, self).__init__()
        # self.wce_loss = Weight_CrossEntropy_Loss(weight=[0.9, 0.1], balance_idx=0)
        self.wce_loss = Weight_CrossEntropy_Loss()
        self.dice = SoftDiceLoss()
        self.gamma = gamma

    def forward(self, net_out, target):
        net_out = F.softmax(net_out, dim=1)
        dice_loss = self.dice(net_out, target)      # weight=0.8
        wce_loss = self.wce_loss(net_out, target)   # weight=0.2
        explog_loss = 0.8 * torch.pow(dice_loss, self.gamma) + 0.2 * wce_loss

        return explog_loss


# if __name__ == '__main__':
#     img = torch.tensor(
#         [[[[0.2, 0.2, 0.3, 0.3],
#            [0.2, 0.2, 0.3, 0.3],
#            [0.2, 0.2, 0.3, 0.3],
#            [0.2, 0.2, 0.3, 0.3]],
#
#           [[0.8, 0.8, 0.7, 0.7],
#            [0.8, 0.8, 0.7, 0.7],
#            [0.8, 0.8, 0.7, 0.7],
#            [0.8, 0.8, 0.7, 0.7]]]]
#     )
#     target = torch.tensor([[[1, 1, 0, 0],
#                             [1, 1, 0, 0],
#                             [1, 1, 0, 0],
#                             [1, 1, 0, 0]]])
#     net = ExpLog_Loss()
#     out = tp_tn_fp_fn(img, target)
#     print(out)


