import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import transforms

from networks.EffTransUNet_seg import EffTransUNet as ViT_seg
from networks.EffTransUNet_seg import CONFIGS as CONFIGS_ViT_seg
from utils import DiceLoss
import requests

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',   type=str,   default='./data/Synapse/train_npz')
    parser.add_argument('--list_dir',    type=str,   default='./lists/lists_Synapse')
    parser.add_argument('--dataset',     type=str,   default='Synapse')
    parser.add_argument('--num_classes', type=int,   default=9)#2,9
    parser.add_argument('--max_epochs1', type=int,   default=300,   help='Phase1')
    parser.add_argument('--max_epochs2', type=int,   default=200,   help='Phase2')
    parser.add_argument('--batch_size',  type=int,   default=24)
    parser.add_argument('--n_gpu',       type=int,   default=1)
    parser.add_argument('--seed',        type=int,   default=20)
    parser.add_argument('--base_lr',     type=float, default=0.001)#0.01,0.001
    parser.add_argument('--img_size',    type=int,   default=224)
    parser.add_argument('--n_skip',      type=int,   default=3)
    parser.add_argument('--vit_name',    type=str,   default='EfficientNet-B3')#EfficientNet-B3,EfficientNet-B4,EfficientNet-B5
    args = parser.parse_args()
    return args


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark   = False


def get_dataloader(args):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    db = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split='train',
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])])
    )
    worker_fn = lambda wid: random.seed(args.seed + wid)
    loader = DataLoader(
        db, batch_size=args.batch_size * args.n_gpu,
        shuffle=True, num_workers=0, pin_memory=True
    )
    return loader


def build_model(args):
    cfg = CONFIGS_ViT_seg[args.vit_name]
    cfg.n_classes = args.num_classes
    cfg.n_skip    = args.n_skip
    backbone_name = args.vit_name.lower().replace('-', '_')
    print(backbone_name)
    model = ViT_seg(args.vit_name, backbone_name=backbone_name, img_size=args.img_size, num_classes=args.num_classes)
    return model.cuda()


def train_phase(model, dataloader, optimizer, scheduler, ce_loss, dice_loss, epochs, writer, start_iter=0):
    model.train()
    iter_num = start_iter
    for ep in range(epochs):
        running = 0.0
        for batch in tqdm(dataloader, desc=f'Epoch {ep+1}/{epochs}', ncols=80):
            imgs = batch['image'].cuda()
            labs = batch['label'].cuda()
            outs = model(imgs)

            loss_ce   = ce_loss(outs, labs.long())
            loss_dice = dice_loss(outs, labs, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            lr = optimizer.param_groups[-1]['lr']
            iter_num += 1
            writer.add_scalar('train/lr', lr, iter_num)
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            running += loss.item()
        avg = running / len(dataloader)
        if (ep+1)%5==0:
            torch.save(model.state_dict(), f'./model/efficient-b5_lr0.001_epoch_300_200/b5_phase2_300_200_lr0.001_epoch{ep+1}.pth')

        logging.info(f"Epoch {ep+1}/{epochs}  avg_loss={avg:.4f}  lr={lr:.2E}")
    return iter_num


def main():
    logging.basicConfig(filename="./model/efficientb3_phase2_300_200_lr0.001_heart_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    args = parse_args()
    setup_seed(args.seed)

    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter(f"runs/{args.vit_name}_TU")

    trainloader = get_dataloader(args)
    model = build_model(args)

    ce_loss   = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    # === Phase1: ===
    for p in model.transformer.embeddings.backbone.net.parameters():
        p.requires_grad = False

    backbone_params = model.transformer.embeddings.backbone.net.parameters()
    head_params     = [p for p in model.parameters() if p.requires_grad]
    optimizer1 = optim.AdamW([
        {'params': backbone_params, 'lr': args.base_lr * 0.1},
        {'params': head_params,     'lr': args.base_lr},
    ], weight_decay=1e-4)
    total_steps1 = args.max_epochs1 * len(trainloader)
    scheduler1 = optim.lr_scheduler.OneCycleLR(
        optimizer1, max_lr=args.base_lr,
        steps_per_epoch=len(trainloader),
        epochs=args.max_epochs1,
        pct_start=0.1, div_factor=25.0, final_div_factor=1e4
    )
    logging.info("Start Phase1: freeze")
    iter_num = train_phase(
        model, trainloader, optimizer1, scheduler1,
        ce_loss, dice_loss, args.max_epochs1, writer, start_iter=0
    )


    # === Phase2:  ===
    for p in model.transformer.embeddings.backbone.net.parameters():
        p.requires_grad = True

    optimizer2 = optim.AdamW([
        {'params': backbone_params, 'lr': args.base_lr * 0.1},
        {'params': model.parameters(), 'lr': args.base_lr},
    ], weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.OneCycleLR(
        optimizer2, max_lr=args.base_lr,
        steps_per_epoch=len(trainloader),
        epochs=args.max_epochs2,
        pct_start=0.0, div_factor=25.0, final_div_factor=1e4
    )
    logging.info("Start Phase2: unfreeze")
    train_phase(
        model, trainloader, optimizer2, scheduler2,
        ce_loss, dice_loss, args.max_epochs2, writer, start_iter=iter_num
    )

    writer.close()
    logging.info("Training Complete.")


if __name__ == '__main__':
    main()
