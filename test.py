import argparse
import logging
import torch
from torch import nn

from models.vit import ViT

from utils import (
    create_dataloaders,
    test_model,
    get_model_configs,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def arugment_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Testing')
    parser.add_argument('--model', default='vit_tiny', type=str, help='model name')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--split', default=0.8, type=float, help='train & validation split size')
    parser.add_argument('--dir', default='./model_checkpoints/ViT_checkpoints/', type=str, help='model checkpoint directory')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--history_path', default='./history/vit_history', type=str, help='training history path')
    return parser


if __name__ == "__main__":
    args = arugment_parser().parse_args()

    seed = args.seed
    checkpoint_dir = args.dir
    image_size = (32,32)

    model_configs = get_model_configs(args.model)

    if args.model == 'vit_tiny':
        model = ViT(
            image_size=image_size,
            num_classes=100,
            **model_configs,
        )
    else:
        model = ViT(
            image_size=image_size,
            num_classes=100,
            **model_configs,
        )

    _, _, test_dataloader = create_dataloaders(
        batch_size=args.bs,
        split=args.split,
        seed=seed,
    )

    criterion = nn.CrossEntropyLoss()

    logging.info('Model Testing Started')

    test_model(
        model=model,
        test_loader=test_dataloader,
        checkpoint_dir=checkpoint_dir,
        criterion=criterion,
    )