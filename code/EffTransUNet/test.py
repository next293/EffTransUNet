import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.EffTransUNet_seg import EffTransUNet as ViT_seg
from networks.EffTransUNet_seg import CONFIGS as CONFIGS_ViT_seg
import time

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse/test_vol_h51', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir1', type=str,
                    default='./lists/lists_Synapse1', help='list dir')

parser.add_argument('--max_iterations', type=int,default=100000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')#epoch,200,150
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=False, action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='EfficientNet-B3', help='select one vit model')#EfficientNet-B3,EfficientNet-B4,EfficientNet-B5

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')# 0.1,0.01,0.001
parser.add_argument('--seed', type=int, default=20, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    #logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    times=[]
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            image, label = sampled_batch["image"], sampled_batch["label"]
            case_name = sampled_batch['case_name'][0]

            # <<< timing start
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            # <<< timing start

            metric_i = test_single_volume(
                image, label, model,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=test_save_path,
                case=case_name, z_spacing=args.z_spacing
            )

            # <<< timing end
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            t_cost = t_end - t_start
            times.append(t_cost)
            print(f"Case {case_name} inference time: {t_cost * 1000:.2f} ms")
            # <<< timing end

            metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    avg_time = sum(times) / len(times)
    print(f"Average inference time per case: {avg_time * 1000:.2f} ms")
    return performance, mean_hd95 #"Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': './data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}".format(args.exp)#snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + 'Eff-b3'+'_heart'
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    # if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    backbone_name = args.vit_name.lower().replace('-', '_')
    print(backbone_name)
    net = ViT_seg(args.vit_name, backbone_name=backbone_name,img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # test weight files，the name of weight file
    weight_files=['./model/efficient-b3_lr0.01_epoch_300_300_CTorgan/b3_phase2_300_300_lr0.01_epoch150.pth']

    for weight_file in weight_files:
        if os.path.exists(weight_file):
            print(f"Testing model: {weight_file}")
        net.load_state_dict(torch.load(weight_file, weights_only=True))
        snapshot_name = snapshot_path.split('/')[-1]

        log_folder = './test_log/test_log_' + args.exp
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        if args.is_savenii:
            args.test_save_dir = './predictions'
            test_save_path = os.path.join(args.test_save_dir, snapshot_name, f"epoch-150")
            os.makedirs(test_save_path, exist_ok=True)
            print(test_save_path)
            print(args.is_savenii)
        else:
            test_save_path = None
        performance, mean_hd95 = inference(args, net, test_save_path)
        print('Testing performance in best val model: mean_dice : %.4f mean_hd95 : %.4f' % (performance, mean_hd95))



