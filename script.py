import argparse
import logging
import torch
from torch import nn

from model import (
    ViT,
)

from utils import (
    create_dataloaders,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def arugment_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--model', default='vit', type=str, help='model name')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer name')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--split', default=0.8, type=float, help='train & validation split size')
    parser.add_argument('--resume', default=False, type=bool, help='resume training?')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--path', default='./model_checkpoints/ViT_checkpoints/', type=str, help='model checkpoint path')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--use_amp', default=True, type=bool, help='whether to use automatic mixed precision')
    parser.add_argument('--continue_training', default=True, type=bool, help='whether to continue training')
    parser.add_argument('--scheduler', default='reduce_lr', type=str, help='learning rate scheduler')
    parser.add_argument('--history_path', default='./history/vit_history', type=str, help='training history path')
    return parser


if __name__ == "__main__":
    args = arugment_parser().parse_args()

    seed = args.seed
    checkpoint_path = args.path
    epochs = args.epochs
    use_amp = args.use_amp
    scheduler = args.scheduler
    history_path = args.history_path
    image_size = (32,32)
    patch_size = (8,8)

    if args.model == 'vit':
        model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=128,
            depth=8, 
            heads=4,
            num_classes=100,
            dropout_rate=0.2,
        )
    else:
        model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=128,
            depth=8, 
            heads=4,
            num_classes=100,
            dropout_rate=0.2,
        )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-4
        )
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=1e-4
        )
    else:
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-4
        )
        logging.info('adam optimizer used')


    if scheduler == 'reduce_lr':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            min_lr=1e-6
        )
    elif scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            args.n_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            args.n_epochs
        )
        logging.info('cosine scheduler used')


    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size=args.bs, split=args.split)

    criterion = nn.CrossEntropyLoss()

    logging.info('Model Training Started')

    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        epochs=epochs,
        criterion=criterion,
        use_amp=use_amp,
        seed=seed,
        patience=15,
        threshold=1e-4,
    )