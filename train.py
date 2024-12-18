import argparse
import logging
import torch
from torch import nn

from models.vit import ViT

from utils import (
    create_dataloaders,
    train_model,
    get_model_configs,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def arugment_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--model', default='vit_custom', type=str, help='model name')
    parser.add_argument('--optim', default='adamw', type=str, help='optimizer name')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--split', default=0.8, type=float, help='train & validation split size')
    parser.add_argument('--resume', default=False, type=bool, help='resume training?')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--dir', default='./model_checkpoints/ViT_checkpoints/', type=str, help='model checkpoint directory')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--use_amp', default=True, type=bool, help='whether to use automatic mixed precision')
    parser.add_argument('--continue_training', default=True, type=bool, help='whether to continue training')
    parser.add_argument('--scheduler', default='cosine', type=str, help='learning rate scheduler')
    parser.add_argument('--drop', default=0.2, type=float, help='drop out rate')
    parser.add_argument('--history_path', default='./history/vit_history.txt', type=str, help='training history path')
    return parser


if __name__ == "__main__":
    args = arugment_parser().parse_args()

    seed = args.seed
    checkpoint_dir = args.dir
    epochs = args.epochs
    use_amp = args.use_amp
    scheduler = args.scheduler
    history_path = args.history_path
    dropout_rate = args.drop
    image_size = (32,32)

    model_configs = get_model_configs(args.model)

    if args.model in ['vit_custom', 'vit_tiny', 'vit_small', 'vit_base']:
        model = ViT(
            image_size=image_size,
            num_classes=100,
            dropout_rate=dropout_rate,
            **model_configs,
        )
    else:
        model = ViT(
            image_size=image_size,
            num_classes=100,
            dropout_rate=dropout_rate,
            **model_configs,
        )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr
        )
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.lr
        )
    elif args.optim == 'adamw':

        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad: 
                continue
            if "bias" in name or "LayerNorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=args.lr
        )
    else:
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr
        )
        logging.info('adam optimizer used')


    if scheduler == 'reduce_lr':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            min_lr=1e-6,
            patience=10,
            verbose=True,
        )
    elif scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            args.epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            args.epochs
        )
        logging.info('cosine scheduler used')


    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        batch_size=args.bs,
        split=args.split,
        seed=seed,
    )

    criterion = nn.CrossEntropyLoss()

    logging.info('Model Training Started')

    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        checkpoint_dir=checkpoint_dir,
        history_path=history_path,
        epochs=epochs,
        criterion=criterion,
        use_amp=use_amp,
        seed=seed,
        patience=15,
        threshold=1e-4,
    )