import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_brain import BrainTumorDataset
from networks.EffTransUNet_cls import EffTransUNet_cls as ViT_seg
from networks.EffTransUNet_cls import CONFIGS as CONFIGS_ViT_seg
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='./data/BrainTumor/test',
                    help='root')
parser.add_argument('--dataset', type=str,
                    default='BrainTumor',
                    help='dataset')
parser.add_argument('--num_classes', type=int,
                    default=4, help='num class')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch size')
parser.add_argument('--img_size', type=int,
                    default=224, help='img size')
parser.add_argument('--n_skip', type=int,
                    default=0, help='skip')
parser.add_argument('--vit_name', type=str,
                    default='EfficientNet-B3',
                    help='backbone')
parser.add_argument('--save_dir', type=str,
                    default='./results',
                    help='save root')
parser.add_argument('--plot_confusion', action='store_true',
                    help='if create confusion_matrix')

args = parser.parse_args()


def test_classification(args, model):

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    testset = BrainTumorDataset(
        data_dir=args.data_path,
        mode='test',
        transform=test_transform
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4
    )


    os.makedirs(args.save_dir, exist_ok=True)


    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(testloader, desc="test"):
            images = batch['image'].cuda()
            labels = batch['label'].cuda()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )


    result_path = os.path.join(args.save_dir, "classification_report.txt")
    with open(result_path, 'w') as f:
        f.write(report)
    print(f"classification report save to {result_path}")


    if args.plot_confusion:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(args.save_dir, 'confusion_matrix.png'))
        print(f"confusion_matrix save to {os.path.join(args.save_dir, 'confusion_matrix.png')}")


if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(42)


    config = CONFIGS_ViT_seg[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip


    model = ViT_seg(args.vit_name, img_size=args.img_size, num_classes=args.num_classes).cuda()


    args.model_path = "./model/CLS_BrainTumor_224_bs24_lr0.0001/epoch_300.pth"
    state_dict = torch.load(args.model_path)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


    test_classification(args, model)