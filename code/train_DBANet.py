import os
import sys
import logging
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_40p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_40p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('-g', '--gpu', type=str, default='1')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numbers
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from skimage import segmentation as skimage_seg

from models.vnet import VNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, print_func, kaiming_normal_init_weight
from utils.softlabel_maxAB_loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config
from utils.dynamice_patch_based_merge_gaussian import merge_patches

config = Config(args.task)



def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size),
                RandomFlip_LR(),
                RandomFlip_UD(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=True
    )

    return model, optimizer


class BDA:
    def __init__(self, num_cls, momentum=0.95):
        self.num_cls = num_cls
        self.momentum = momentum

    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        P = (num_each_class.max()+1e-8) / (num_each_class+1e-8)
        P_log = torch.log(P)
        if P_log.max() == 0:
            weight = torch.ones_like(P_log)  
        else:
            weight = P_log / P_log.max()  
        return weight

    def compute_edge_pixel(self, img_gt, out_shape):
        edges = torch.zeros_like(img_gt).cuda()
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)
        out_shape = [config.num_cls] + list(out_shape)
        # Iterate over each class (excluding the background class 0)
        for c in range(out_shape[0]):
            posmask = (img_gt == c).astype(np.bool_)
            if posmask.any():
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                edges = torch.logical_or(edges, torch.tensor(boundary).cuda())
        return edges

    def init_weights(self, labeled_dataset):
        if labeled_dataset.unlabeled:
            raise ValueError
        num_each_class = np.zeros(self.num_cls)
        edge_each_class = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list:
            _, _, label = labeled_dataset._get_data(data_id)
            label = torch.tensor(label).cuda()
            edge_voxels = self.compute_edge_pixel(label, label.shape)
            label = label.cpu().numpy()
            for cls in range(self.num_cls):
                num_each_class[cls] += (label == cls).sum()
                edge_each_class[cls] += edge_voxels[label == cls].sum().item()
        weights_edge = self._cal_weights(edge_each_class)
        #weights_non_edge = self._cal_weights(num_each_class - edge_each_class)
        self.weights = weights_edge * self.num_cls
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        label_numpy = pseudo_label.data.cpu().numpy()
        num_each_class = np.zeros(self.num_cls)
        edge_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = torch.tensor(label_numpy[i]).cuda().squeeze()
            edge_voxels = self.compute_edge_pixel(label, label.shape)
            label = label.cpu().numpy()
            for cls in range(self.num_cls):
                num_each_class[cls] += (label == cls).sum()
                edge_each_class[cls] += edge_voxels[label == cls].sum().item()
        weights_edge = self._cal_weights(edge_each_class)
        #weights_non_edge = self._cal_weights(num_each_class - edge_each_class)
        cur_weights = weights_edge * self.num_cls
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights


class BoundaryLoss(nn.Module):
    """Boundary Loss for 3D Segmentation"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        n, c, _, _, _ = pred.shape
        pred = torch.softmax(pred, dim=1)
        #one_hot_gt = gt

        # Boundary map for ground truth
        gt_b = F.max_pool3d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2
        )
        gt_b -= 1 - gt
        # Boundary map for predictions
        pred_b = F.max_pool3d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2
        )
        pred_b -= 1 - pred
        # Extended boundary map for ground truth
        gt_b_ext = F.max_pool3d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )
        # Extended boundary map for predictions
        pred_b_ext = F.max_pool3d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )

        # Reshape to (N, C, -1)
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision and Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)
        # Summing BF1 Score for each class and averaging over mini-batch
        loss = torch.mean(BF1, dim=0)

        return loss


class BVA:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.last_loss = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.boundloss_func = BoundaryLoss()
        self.cls_learn_dice = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn_dice = torch.zeros(num_cls).float().cuda()
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.loss_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(config.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights


    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_loss = self.boundloss_func(pred, y_onehot)
        cur_loss = torch.where(cur_loss == 0, torch.tensor(1e-8, device=cur_loss.device), cur_loss)
        delta_loss = cur_loss - self.last_loss
        cur_cls_learn = torch.where(delta_loss>0, delta_loss, torch.tensor(0.0, dtype=delta_loss.dtype).cuda()) * torch.log(cur_loss / self.last_loss)
        cur_cls_unlearn = torch.where(delta_loss<=0, delta_loss, torch.tensor(0.0, dtype=delta_loss.dtype).cuda()) * torch.log(cur_loss / self.last_loss)
        self.last_loss = cur_loss
        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(cur_diff, 1/5)
        self.loss_weight = EMA(1.-cur_loss, self.loss_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.loss_weight
        weights = weights / weights.max()
        return weights * self.num_cls



class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.kernel_size = kernel_size

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        # Calculate padding size for "same" padding
        if self.conv == F.conv1d:
            pad_size = self.kernel_size[0] // 2
            padding = (pad_size,)
        elif self.conv == F.conv2d:
            pad_size = [k // 2 for k in self.kernel_size]
            padding = (pad_size[1], pad_size[1], pad_size[0], pad_size[0])
        elif self.conv == F.conv3d:
            pad_size = [k // 2 for k in self.kernel_size] 
            padding = (pad_size[2], pad_size[2], pad_size[1], pad_size[1], pad_size[0], pad_size[0]) 
        
        input = F.pad(input, padding) 
        return self.conv(input, weight=self.weight.cuda(), groups=self.groups)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def stable_gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    gumbels = sample_gumbel(logits.shape)
    y = logits + gumbels
    y = F.softmax(y / tau, dim=dim)
    if hard:
        _, k = y.max(dim=dim)
        y_hard = torch.zeros_like(y).scatter_(dim, k.unsqueeze(dim), 1.0)
        y = y_hard - y.detach() + y
    return y


if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B  = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = kaiming_normal_init_weight(model_B)


    # make loss function
    bvadw = BVA(config.num_cls, accumulate_iters=50)
    bdadw = BDA(config.num_cls, momentum=0.99)

    weight_A = bvadw.init_weights()
    weight_B = bdadw.init_weights(labeled_loader.dataset)

    
    loss_func_A     = make_loss_function(args.sup_loss, weight_A)
    loss_func_B     = make_loss_function(args.sup_loss, weight_B)
    cps_loss_func_A = make_loss_function(args.cps_loss, weight_A)
    cps_loss_func_B = make_loss_function(args.cps_loss, weight_B)


    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    patch_size = (4, 4, 4)

    # Start timing the entire training process
    start_time = time.time()
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model_A.train()
        model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():
                    output_A = model_A(image)
                    output_B = model_B(image)
                    del image

                    # sup (ce + dice)
                    output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                    output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]

                    with torch.no_grad():
                        smoothing = GaussianSmoothing(config.num_cls, 3, 1)
                        output_A1 = smoothing(stable_gumbel_softmax(output_A, dim=1))
                        outputA = output_A1 + output_B
                        output_B1 = smoothing(stable_gumbel_softmax(output_B, dim=1))
                        outputB = output_B1 + output_A
     
                    mergeAB_U = merge_patches(outputA[tmp_bs:, ...], outputB[tmp_bs:, ...], patch_size)
                    mergeA = torch.cat([outputA[:tmp_bs, ...], mergeAB_U], dim=0)
                    mergeB = torch.cat([outputB[:tmp_bs, ...], mergeAB_U], dim=0)
                    soft_A = F.softmax(mergeA.detach(), dim=1)
                    soft_B = F.softmax(mergeB.detach(), dim=1)
                    
                    weight_A = bvadw.cal_weights(output_A_l.detach(), label_l.detach())
                    weight_B = bdadw.get_ema_weights(mergeAB_U.detach())

                    loss_func_A.update_weight(weight_A)
                    loss_func_B.update_weight(weight_B)
                    cps_loss_func_A.update_weight(weight_A)
                    cps_loss_func_B.update_weight(weight_B)

                    loss_sup = loss_func_A(output_A_l, label_l) + loss_func_B(output_B_l, label_l)
                    loss_cps = cps_loss_func_A(output_A,soft_B) + cps_loss_func_B(output_B,soft_A) 
                    loss = loss_sup + cps_w * loss_cps


                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()
                # if epoch_num>0:

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())

        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        # print(dict(zip([i for i in range(config.num_cls)] ,print_func(weight_A))))
        writer.add_scalars('class_weights/A', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_A))), epoch_num)
        writer.add_scalars('class_weights/B', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_B))), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')
        # logging.info(f'     cps_w: {cps_w}')
        # if epoch_num>0:
        logging.info(f"     Class Weights A: {print_func(weight_A)}, lr: {get_lr(optimizer_A)}")
        logging.info(f"     Class Weights B: {print_func(weight_B)}")
        # logging.info(f"     Class Weights u: {print_func(weight_u)}")
        # lr_scheduler_A.step()
        # lr_scheduler_B.step()
        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        # print(optimizer_A.param_groups[0]['lr'])
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    output = (model_A(image) + model_B(image))/2.0
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')

            # '''
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': model_A.state_dict(),
                    'B': model_B.state_dict()
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break
    writer.close()