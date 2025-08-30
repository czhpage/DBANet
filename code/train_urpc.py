import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='cps')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=False) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.unet_ds import unet_3D_ds
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from utils.loss import SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config
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


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


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
    model = unet_3D_ds(n_classes=config.num_cls, in_channels=1).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    return model, optimizer




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
    model, optimizer = make_model_all()
    # model_B, optimizer_B = make_model_all()
    # model_A = kaiming_normal_init_weight(model_A)
    # model_B = xavier_normal_init_weight(model_B)

    logging.info(optimizer)

    # make loss function
    # weight,_ = labeled_loader.dataset.weight()
    # print(weight)
    # print(sum(weight))

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = SoftDiceLoss(smooth=1e-8)
    kl_distance = nn.KLDivLoss(reduction='none')



    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model.train()
        # model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer.zero_grad()
            # optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():


                    outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4,  = model(
                        image)
                    outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
                    outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
                    outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
                    outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)

                    label_l = label_l.squeeze(1)

                    loss_ce_aux1 = ce_loss(outputs_aux1[:tmp_bs],
                                           label_l[:tmp_bs])
                    loss_ce_aux2 = ce_loss(outputs_aux2[:tmp_bs],
                                           label_l[:tmp_bs])
                    loss_ce_aux3 = ce_loss(outputs_aux3[:tmp_bs],
                                           label_l[:tmp_bs])
                    loss_ce_aux4 = ce_loss(outputs_aux4[:tmp_bs],
                                           label_l[:tmp_bs])

                    loss_dice_aux1 = dice_loss(
                        outputs_aux1_soft[:tmp_bs], label_l[:tmp_bs].unsqueeze(1))
                    loss_dice_aux2 = dice_loss(
                        outputs_aux2_soft[:tmp_bs], label_l[:tmp_bs].unsqueeze(1))
                    loss_dice_aux3 = dice_loss(
                        outputs_aux3_soft[:tmp_bs], label_l[:tmp_bs].unsqueeze(1))
                    loss_dice_aux4 = dice_loss(
                        outputs_aux4_soft[:tmp_bs], label_l[:tmp_bs].unsqueeze(1))

                    supervised_loss = (loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4 +
                                       loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4)/8

                    preds = (outputs_aux1_soft +
                             outputs_aux2_soft+outputs_aux3_soft+outputs_aux4_soft)/4

                    variance_aux1 = torch.sum(kl_distance(
                        torch.log(outputs_aux1_soft[tmp_bs:]), preds[tmp_bs:]), dim=1, keepdim=True)
                    exp_variance_aux1 = torch.exp(-variance_aux1)

                    variance_aux2 = torch.sum(kl_distance(
                        torch.log(outputs_aux2_soft[tmp_bs:]), preds[tmp_bs:]), dim=1, keepdim=True)
                    exp_variance_aux2 = torch.exp(-variance_aux2)

                    variance_aux3 = torch.sum(kl_distance(
                        torch.log(outputs_aux3_soft[tmp_bs:]), preds[tmp_bs:]), dim=1, keepdim=True)
                    exp_variance_aux3 = torch.exp(-variance_aux3)

                    variance_aux4 = torch.sum(kl_distance(
                        torch.log(outputs_aux4_soft[tmp_bs:]), preds[tmp_bs:]), dim=1, keepdim=True)
                    exp_variance_aux4 = torch.exp(-variance_aux4)


                    consistency_dist_aux1 = (
                                                    preds[tmp_bs:] - outputs_aux1_soft[tmp_bs:]) ** 2
                    consistency_loss_aux1 = torch.mean(
                        consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

                    consistency_dist_aux2 = (
                                                    preds[tmp_bs:] - outputs_aux2_soft[tmp_bs:]) ** 2
                    consistency_loss_aux2 = torch.mean(
                        consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

                    consistency_dist_aux3 = (
                                                    preds[tmp_bs:] - outputs_aux3_soft[tmp_bs:]) ** 2
                    consistency_loss_aux3 = torch.mean(
                        consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

                    consistency_dist_aux4 = (
                                                    preds[tmp_bs:] - outputs_aux4_soft[tmp_bs:]) ** 2
                    consistency_loss_aux4 = torch.mean(
                        consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)

                    consistency_loss = (consistency_loss_aux1 +
                                        consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4
                    loss = supervised_loss + cps_w * consistency_loss


                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                # amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(supervised_loss.item())
            loss_cps_list.append(consistency_loss.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}, cpsw:{cps_w} lr: {get_lr(optimizer)}')


        # lr_scheduler_A.step()
        # lr_scheduler_B.step()

        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    # output = model_A(image)
                    output = model(image)
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
                torch.save(model.state_dict(), save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')

            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()
