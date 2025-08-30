import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='fully')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str,  default='0')
parser.add_argument('--cps', type=str, default=None)
# EVIL specific parameters
parser.add_argument('--temperature', type=float, default=0.25, help='temperature for evidence calculation')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from models.vnet import VNet, VNet4SSNet
from models.vnet_dst import VNet_Decoupled
from models.unet_ds import unet_3D_ds
from models.unet import unet_3D
from models.unet_3d import UNet3D  # 3D UNet
from utils import test_all_case, read_list, maybe_mkdir, test_all_case_AB
from utils.config import Config
config = Config(args.task)

if __name__ == '__main__':
    stride_dict = {
        0: (32, 16),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]

    snapshot_path = f'./logs/{args.exp}/'
    test_save_path = f'./logs/{args.exp}/predictions_{args.cps}/'
    maybe_mkdir(test_save_path)

    if "fully" in args.exp:
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model.eval()
        args.cps = None


    elif "dst" in args.exp:
        model_A = VNet_Decoupled(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model_B = VNet_Decoupled(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model_A.eval()
        model_B.eval()

    elif "urpc" in args.exp:
        model = unet_3D_ds(n_classes=config.num_cls, in_channels=1).cuda()
        model.eval()
        args.cps = None
    # elif "acisis" in args.exp:
    #     model = unet_3D(n_classes=config.num_cls, in_channels=1).cuda()
    #     model.eval()
    #     args.cps = None

    elif "uamt" in args.exp or "acisis" in args.exp:
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model.eval()
        args.cps = None
    elif "ssnet" in args.exp:
        model = VNet4SSNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False).cuda()
        model.eval()
        args.cps = None
    elif "evil" in args.exp:
        unet_params = {
            'in_chns': config.num_channels,
            'feature_chns': [16, 32, 64, 128, 256], 
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': config.num_cls,
            'bilinear': False,
            'acti_func': 'relu'
        }
        
        # Evi-Net: Evidence-based network
        model_A = UNet3D(
            in_chns=unet_params['in_chns'],
            class_num=unet_params['class_num'],
            feature_chns=unet_params['feature_chns'],
            dropout=unet_params['dropout'],
            bilinear=unet_params['bilinear']
        ).cuda()
        
        # Seg-Net: Standard segmentation network
        model_B = UNet3D(
            in_chns=unet_params['in_chns'],
            class_num=unet_params['class_num'],
            feature_chns=unet_params['feature_chns'],
            dropout=unet_params['dropout'],
            bilinear=unet_params['bilinear']
        ).cuda()
        
        model_A.eval()
        model_B.eval()
    else:
        model_A = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model_B = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model_A.eval()
        model_B.eval()


    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')



    with torch.no_grad():
        if args.cps == "AB":
            model_A.load_state_dict(torch.load(ckpt_path)["A"])
            model_B.load_state_dict(torch.load(ckpt_path)["B"])
            print(f'load checkpoint from {ckpt_path}')
            test_all_case_AB(
                model_A, model_B,
                read_list(args.split, task=args.task),
                task=args.task,
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                stride_z=stride[1],
                test_save_path=test_save_path
            )
        else:
            if args.cps:
                model.load_state_dict(torch.load(ckpt_path)[args.cps])
            else: # for full-supervision
                model.load_state_dict(torch.load(ckpt_path))
            print(f'load checkpoint from {ckpt_path}')
            test_all_case(
                model,
                read_list(args.split, task=args.task),
                task=args.task,
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                stride_z=stride[1],
                test_save_path=test_save_path
            )
