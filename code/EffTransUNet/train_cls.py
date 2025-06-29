import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import transforms

from networks.EffTransUNet_cls import EffTransUNet_cls as ViT_seg
from networks.EffTransUNet_cls import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_brain import BrainTumorDataset  # 分类数据集

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/BrainTumor/train', help='root')
parser.add_argument('--dataset', type=str,
                    default='BrainTumor', help='dataset')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_BrainTumor', help='（list')
parser.add_argument('--num_classes', type=int,
                    default=4, help='num')
parser.add_argument('--max_iterations', type=int,
                    default=100000, help='max_iterations')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='max')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch size')
parser.add_argument('--n_gpu', type=int, default=1, help='GPU number')
parser.add_argument('--deterministic', type=int, default=1,
                    help='deterministic')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='basic lr')
parser.add_argument('--img_size', type=int,
                    default=224, help='size')
parser.add_argument('--seed', type=int,
                    default=42, help='seed')
parser.add_argument('--n_skip', type=int,
                    default=0, help='skip')
parser.add_argument('--vit_name', type=str,
                    default='EfficientNet-B3', help='backbone')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='ViT-patch_size')
parser.add_argument('--max_epochs1', type=int, default=150, help='Phase1 epoch')
parser.add_argument('--max_epochs2', type=int, default=150, help='Phase2 epoch')
args = parser.parse_args()


def trainer_classifier(args, model, snapshot_path):
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    db_train = BrainTumorDataset(args.root_path, mode='train', transform=train_transform)
    print("num of data:", len(db_train))
    def worker_init_fn(wid): random.seed(args.seed + wid)
    trainloader = DataLoader(db_train,
                             batch_size=args.batch_size*args.n_gpu,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu>1:
        model = nn.DataParallel(model)

    model = model.cuda()
    model.train()
    criterion = CrossEntropyLoss()
    phase1 = args.max_epochs1
    phase2 = args.max_epochs2
    total_epochs = phase1 + phase2
    iter_num = 0
    best_acc = 0.0

    # --- Phase1
    for p in model.transformer.embeddings.backbone.net.parameters():
        p.requires_grad = False

    vit_params = [p for p in model.transformer.parameters() if p.requires_grad]
    head_params = list(model.fc.parameters())

    optimizer = optim.AdamW([
        {'params': vit_params, 'lr': args.base_lr},
        {'params': head_params, 'lr': args.base_lr}
    ], weight_decay=1e-4)

    scheduler1 = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.base_lr,
        steps_per_epoch=len(trainloader),
        epochs=phase1,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1e4
    )

    logging.info(f"==> Phase1: Freeze backbone, train ViT & Head for {phase1} epochs")
    for epoch in range(phase1):
        running_loss, correct, total_n = 0.0, 0, 0

        lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        logging.info(f"Phase1 Epoch {epoch + 1}/{phase1} LRs: {lrs}")

        for batch in tqdm(trainloader, desc=f"P1 E{epoch + 1}/{phase1}", ncols=80):
            imgs = batch['image'].cuda()
            labs = batch['label'].cuda()
            logits = model(imgs)
            loss = criterion(logits, labs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler1.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total_n += labs.size(0)
            iter_num += 1

        acc = 100. * correct / total_n
        logging.info(f"Phase1 {epoch + 1}/{phase1} loss={running_loss / len(trainloader):.4f} acc={acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best.pth'))

        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(snapshot_path, f'epoch_{epoch + 1}.pth'))

    # --- Phase2
    for p in model.transformer.embeddings.backbone.net.parameters():
        p.requires_grad = True

    backbone_params = list(model.transformer.embeddings.backbone.net.parameters())
    vit_params = [p for p in model.transformer.parameters() if not any(p is bp for bp in backbone_params)]
    head_params = list(model.fc.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.base_lr * 0.1},
        {'params': vit_params, 'lr': args.base_lr},
        {'params': head_params, 'lr': args.base_lr}
    ], weight_decay=1e-4)

    scheduler2 = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.base_lr * 0.1, args.base_lr, args.base_lr],
        steps_per_epoch=len(trainloader),
        epochs=phase2,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4
    )

    logging.info(f"==> Phase2: Unfreeze backbone, fine-tune all for {phase2} epochs")
    for epoch in range(phase2):
        running_loss, correct, total_n = 0.0, 0, 0
        lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        logging.info(f"Phase2 Epoch {epoch + 1}/{phase2} LRs: {lrs}")

        for batch in tqdm(trainloader, desc=f"P2 E{epoch + 1}/{phase2}", ncols=80):
            imgs = batch['image'].cuda()
            labs = batch['label'].cuda()
            logits = model(imgs)
            loss = criterion(logits, labs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler2.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total_n += labs.size(0)
            iter_num += 1

        global_epoch = phase1 + epoch + 1
        acc = 100. * correct / total_n
        logging.info(f"Epoch {global_epoch}/{total_epochs} loss={running_loss / len(trainloader):.4f} acc={acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best.pth'))

        if global_epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(snapshot_path, f'epoch_{global_epoch}.pth'))

    logging.info(f"finish best: {best_acc:.2f}%")
    return "Done"


if __name__=='__main__':

    cudnn.benchmark   = not args.deterministic
    cudnn.deterministic = bool(args.deterministic)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    dataset_config = {
        'BrainTumor':{
            'root_path': './data/BrainTumor/train',
            'list_dir': './lists/lists_BrainTumor',
            'num_classes': 4,
        },
    }
    cfg = dataset_config[args.dataset]
    args.root_path   = cfg['root_path']
    args.list_dir    = cfg.get('list_dir','')
    args.num_classes = cfg['num_classes']


    args.exp = f"CLS_{args.dataset}_{args.img_size}"
    snapshot = os.path.join("./model", args.exp + f"_bs{args.batch_size}_lr{args.base_lr}_skip{args.n_skip}")
    os.makedirs(snapshot, exist_ok=True)


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip    = 0
    config_vit.classifier= 'cls'
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (args.img_size//args.vit_patches_size,)*2
    model = ViT_seg(args.vit_name, img_size=args.img_size, num_classes=args.num_classes).cuda()


    trainer = {'BrainTumor': trainer_classifier}
    trainer[args.dataset](args, model, snapshot)
