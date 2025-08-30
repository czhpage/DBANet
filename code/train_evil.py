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
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--consistency_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
# EVIL specific parameters
parser.add_argument('--temperature', type=float, default=0.3, help='temperature for evidence calculation')
parser.add_argument('--uncertainty_threshold', type=float, default=0.5, help='uncertainty threshold for pseudo label filtering')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.nn as nn

from models.unet_3d import UNet3D  # 3D UNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, kaiming_normal_init_weight
from utils.loss import RobustCrossEntropyLoss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config

config = Config(args.task)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.consistency_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.consistency_w


def kl_divergence_3d(alpha, num_classes, batch_size):
    """3D KL divergence for EDL loss"""
    device = alpha.device
    ones = torch.ones([batch_size, num_classes] + list(alpha.shape[2:]), dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss_3d(func, y, alpha, epoch_num, num_classes, annealing_step):
    """3D EDL loss function"""
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y.float() * (func(S.float()) - func(alpha.float())), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=alpha.device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=alpha.device),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    b = y.shape[0]
    #kl_div = annealing_coef * kl_divergence_3d(kl_alpha, num_classes, batch=b)
    kl_div = annealing_coef * kl_divergence_3d(kl_alpha, num_classes, b)
    return A + kl_div


def edl_digamma_loss_3d(evidence, target, epoch_num, num_classes, annealing_step):
    """3D EDL digamma loss"""
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss_3d(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss


def dice_loss_3d(score, target):
    """3D Dice loss"""
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class DiceLoss3D(nn.Module):
    """3D Dice Loss Module with vectorized computation"""
    def __init__(self, n_classes):
        super(DiceLoss3D, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """
        Efficient one-hot encoding using scatter for input tensor of shape [B, 1, D, H, W]
        Returns tensor of shape [B, C, D, H, W]
        """
        input_tensor = input_tensor.long()  # Ensure it's integer type
        shape = list(input_tensor.shape)  # [B, 1, D, H, W]
        shape[1] = self.n_classes           # Replace channel dim with n_classes

        one_hot = torch.zeros(shape, device=input_tensor.device)
        one_hot.scatter_(1, input_tensor, 1)  # input_tensor must be [B, 1, D, H, W]
        return one_hot

    def _dice_loss_vectorized(self, score, target, weight=None):
        """
        Vectorized dice loss computation for all classes simultaneously
        Args:
            score: [B, C, D, H, W] - predicted probabilities
            target: [B, C, D, H, W] - one-hot encoded ground truth
            weight: [C] - class weights (optional)
        Returns:
            dice_loss: scalar - weighted average dice loss across all classes
        """
        target = target.float()
        smooth = 1e-5
        
        # Compute intersection, y_sum, z_sum for all classes at once
        # Sum over spatial dimensions (D, H, W) but keep batch and class dimensions
        intersect = torch.sum(score * target, dim=(2, 3, 4))  # [B, C]
        y_sum = torch.sum(target, dim=(2, 3, 4))              # [B, C]
        z_sum = torch.sum(score, dim=(2, 3, 4))               # [B, C]
        
        # Compute dice coefficient for each class and batch
        dice_coeff = (2 * intersect + smooth) / (z_sum + y_sum + smooth)  # [B, C]
        dice_loss = 1 - dice_coeff  # [B, C]
        
        # Apply class weights if provided
        if weight is not None:
            weight = torch.tensor(weight, device=score.device, dtype=score.dtype)
            if weight.dim() == 1:
                weight = weight.unsqueeze(0)  # [1, C]
            dice_loss = dice_loss * weight    # [B, C]
        
        # Average over batch and classes
        return torch.mean(dice_loss)

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Forward pass with vectorized dice loss computation
        Args:
            inputs: [B, C, D, H, W] - network predictions
            target: [B, 1, D, H, W] - ground truth labels
            weight: list or tensor - class weights (optional)
            softmax: bool - whether to apply softmax to inputs
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        target_onehot = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target_onehot.size(), 'predict & target shape do not match'
        loss = self._dice_loss_vectorized(inputs, target_onehot, weight)
        return loss


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
    """Create EVIL models (Evi-Net and Seg-Net) using 3D UNet architecture"""
    # EVIL parameters
    unet_params = {
        'in_chns': config.num_channels,
        'feature_chns': [16, 32, 64, 128, 256], 
        'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
        'class_num': config.num_cls,
        'bilinear': False,
        'acti_func': 'relu'
    }
    
    # Evi-Net: Evidence-based network 
    evi_net = UNet3D(
        in_chns=unet_params['in_chns'],
        class_num=unet_params['class_num'],
        feature_chns=unet_params['feature_chns'],
        dropout=unet_params['dropout'],
        bilinear=unet_params['bilinear']
    ).cuda()
    
    # Seg-Net: Standard segmentation network 
    seg_net = UNet3D(
        in_chns=unet_params['in_chns'],
        class_num=unet_params['class_num'],
        feature_chns=unet_params['feature_chns'],
        dropout=unet_params['dropout'],
        bilinear=unet_params['bilinear']
    ).cuda()
    
    optimizer_evi = optim.SGD(
        evi_net.parameters(),
        lr=args.base_lr,
        momentum=0.9,  
        weight_decay=3e-5,
        nesterov=True
    )
    
    optimizer_seg = optim.SGD(
        seg_net.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return evi_net, seg_net, optimizer_evi, optimizer_seg


def create_onehot_3d(label_batch, num_classes):
    """Create one-hot encoding for 3D labels"""
    batch_size = label_batch.shape[0]
    spatial_shape = label_batch.shape[1:]
    
    onehot = torch.zeros(batch_size, num_classes, *spatial_shape, device=label_batch.device)
    for i in range(num_classes):
        onehot[:, i, ...] = (label_batch == i).float()
    
    return onehot


def create_onehot_3d(label_batch, num_classes):
    """Create one-hot encoding for 3D labels"""
    batch_size = label_batch.shape[0]
    spatial_shape = label_batch.shape[2:]
    onehot = torch.zeros(batch_size, num_classes, *spatial_shape, device=label_batch.device) # [B, 1, D, H, W]
    onehot.scatter_(1, label_batch, 1)
    return onehot


if __name__ == '__main__':
    import random
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # Make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # Make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)

    logging.info(f'{len(labeled_loader)} iterations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} iterations per epoch (unlabeled)')

    # Make EVIL models and optimizers
    evi_net, seg_net, optimizer_evi, optimizer_seg = make_model_all()
    evi_net = kaiming_normal_init_weight(evi_net)
    seg_net = kaiming_normal_init_weight(seg_net)

    # Make loss functions
    ce_loss = RobustCrossEntropyLoss()
    dice_loss = DiceLoss3D(config.num_cls)

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    consistency_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0

    # Start training
    start_time = time.time()
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_evi_list = []
        loss_seg_list = []
        loss_consistency_list = []

        evi_net.train()
        seg_net.train()
        
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_evi.zero_grad()
            optimizer_seg.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():
                    # Forward pass through both networks
                    outputs_evi = evi_net(image)
                    outputs_seg = seg_net(image)
                    
                    # Split labeled and unlabeled data
                    outputs_evi_l, outputs_evi_u = outputs_evi[:tmp_bs], outputs_evi[tmp_bs:]
                    outputs_seg_l, outputs_seg_u = outputs_seg[:tmp_bs], outputs_seg[tmp_bs:]
                    
                    # EVIL: Evidence calculation for Evi-Net
                    evidence_l = torch.exp(torch.tanh(outputs_evi_l) / args.temperature)
                    evidence_u = torch.exp(torch.tanh(outputs_evi_u) / args.temperature)
                    
                    # Calculate alpha parameters (Dirichlet distribution)
                    alpha_l = evidence_l + 1
                    alpha_u = evidence_u + 1
                    
                    # Calculate uncertainty for unlabeled data
                    S_u = torch.sum(alpha_u, dim=1, keepdim=True)
                    uncertainty_u = config.num_cls / S_u
                    
                    # Calculate predictions from evidence
                    belief_l = evidence_l / torch.sum(alpha_l, dim=1, keepdim=True)
                    belief_u = evidence_u / torch.sum(alpha_u, dim=1, keepdim=True)
                    preds_evi_l = F.softmax(belief_l, dim=1)
                    preds_evi_u = F.softmax(belief_u, dim=1)
                    
                    # Standard softmax predictions for Seg-Net
                    preds_seg_l = F.softmax(outputs_seg_l, dim=1)
                    preds_seg_u = F.softmax(outputs_seg_u, dim=1)
                    
                    # Create one-hot encoding for labeled data
                    onehot_l = create_onehot_3d(label_l, config.num_cls)
                    
                    # Supervised losses
                    # Evi-Net: EDL loss + Dice loss
                    edl_loss = edl_digamma_loss_3d(
                        evidence_l, onehot_l, epoch_num, config.num_cls, args.max_epoch // 2
                    )
                    dice_loss_evi = dice_loss(preds_evi_l, label_l, softmax=False)
                    loss_evi_sup = edl_loss + dice_loss_evi
                    
                    # Seg-Net: Standard CE + Dice loss
                    ce_loss_seg = ce_loss(outputs_seg_l, label_l.long())
                    dice_loss_seg = dice_loss(preds_seg_l, label_l, softmax=False)
                    loss_seg_sup = 0.5 * (ce_loss_seg + dice_loss_seg)
                    
                    # Generate pseudo labels
                    pseudo_labels_evi = torch.argmax(preds_evi_u.detach(), dim=1, keepdim=True)
                    pseudo_labels_seg = torch.argmax(preds_seg_u.detach(), dim=1, keepdim=True)
                    
                    # Create one-hot for unlabeled pseudo labels from Seg-Net
                    onehot_pseudo_seg = create_onehot_3d(pseudo_labels_seg, config.num_cls)
                    
                    # Consistency losses with uncertainty-aware filtering
                    # Evi-Net learns from Seg-Net's pseudo labels
                    pseudo_loss_evi = edl_digamma_loss_3d(
                        evidence_u, onehot_pseudo_seg, epoch_num, config.num_cls, args.max_epoch // 2
                    )
                    
                    # Seg-Net learns from Evi-Net's filtered pseudo labels
                    # Apply uncertainty-based filtering
                    uncertainty_mask = (uncertainty_u < args.uncertainty_threshold).float()
                    filtered_pseudo_loss_seg = ce_loss(
                        uncertainty_mask * outputs_seg_u, 
                        (uncertainty_mask.squeeze(1) * pseudo_labels_evi.squeeze(1)).long()
                    )
                    
                    # Total losses
                    loss_evi_total = loss_evi_sup + consistency_w * pseudo_loss_evi
                    loss_seg_total = loss_seg_sup + consistency_w * filtered_pseudo_loss_seg
                    
                    total_loss = loss_evi_total + loss_seg_total

                # Backward pass
                amp_grad_scaler.scale(total_loss).backward()
                amp_grad_scaler.step(optimizer_evi)
                amp_grad_scaler.step(optimizer_seg)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(total_loss.item())
            loss_evi_list.append(loss_evi_total.item())
            loss_seg_list.append(loss_seg_total.item())
            loss_consistency_list.append((pseudo_loss_evi + filtered_pseudo_loss_seg).item())

        # Logging
        writer.add_scalar('lr', get_lr(optimizer_evi), epoch_num)
        writer.add_scalar('consistency_w', consistency_w, epoch_num)
        writer.add_scalar('loss/total_loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/evi_loss', np.mean(loss_evi_list), epoch_num)
        writer.add_scalar('loss/seg_loss', np.mean(loss_seg_list), epoch_num)
        writer.add_scalar('loss/consistency_loss', np.mean(loss_consistency_list), epoch_num)
        
        logging.info(f'epoch {epoch_num} : total_loss : {np.mean(loss_list)}')
        logging.info(f'     evi_loss: {np.mean(loss_evi_list)}, seg_loss: {np.mean(loss_seg_list)}')
        logging.info(f'     consistency_loss: {np.mean(loss_consistency_list)}, consistency_w: {consistency_w}')
        logging.info(f'     lr: {get_lr(optimizer_evi)}')

        # Update learning rate
        optimizer_evi.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_seg.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        consistency_w = get_current_consistency_weight(epoch_num)

        # Evaluation
        if epoch_num % 10 == 0:
            dice_list = [[] for _ in range(config.num_cls-1)]
            evi_net.eval()
            seg_net.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    
                    # Ensemble prediction from both networks
                    output_evi = evi_net(image)
                    output_seg = seg_net(image)
                    
                    # For Evi-Net, convert evidence to predictions
                    evidence = torch.exp(torch.tanh(output_evi) / args.temperature)
                    alpha = evidence + 1
                    belief = evidence / torch.sum(alpha, dim=1, keepdim=True)
                    pred_evi = F.softmax(belief, dim=1)
                    
                    # For Seg-Net, standard softmax
                    pred_seg = F.softmax(output_seg, dim=1)
                    
                    # Ensemble prediction
                    output = (pred_evi + pred_seg) / 2.0
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

            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': evi_net.state_dict(),  # A corresponds to Evi-Net
                    'B': seg_net.state_dict()   # B corresponds to Seg-Net
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()