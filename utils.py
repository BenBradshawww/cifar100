from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from tqdm import tqdm

import torch
import time
import csv
import os


def create_dataloaders(batch_size: int = 32, split: float = 0.8, seed:int = 42):

    torch.manual_seed(seed)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_and_val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=test_and_val_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_and_val_transform)

    train_size = int(split * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader


def write_to_csv(history_path, epoch, train_loss, train_acc, val_loss, val_acc):
    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    row = [epoch + 1, train_loss, train_acc, val_loss, val_acc]
    
    if epoch == 0 and os.path.isfile(history_path):
        os.remove(history_path)
    
    write_header = not os.path.isfile(history_path)
    with open(history_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def clean_history(history_path, epoch_threshold):

    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader if int(row[header.index("epochs")]) <= epoch_threshold]

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)  


def get_last_epoch(history_path):
    if not os.path.isfile(history_path):
        return 0
    
    with open(history_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        last_row = None
        for last_row in reader:
            pass 
    
    return int(last_row[0]) if last_row else 0


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    checkpoint_dir,
    history_path,
    epochs:int,
    criterion,
    use_amp:bool=True,
    seed:int=42,
    patience:int=15,
    threshold:float=1e-4,
    continue_training:bool=True,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    best_loss = float("inf")
    start_time = time.time()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model.to(device)
    offset = 0
    early_stop_counter = 0

    checkpoints = [x for x in os.listdir(checkpoint_dir) if x.endswith('.pth')]
    checkpoints.sort()

    # Load the previous checkpoint if it exists
    if continue_training and checkpoints:
        final_checkpoint = checkpoints[-1]
        
        checkpoint = torch.load(
            final_checkpoint,
            map_location=torch.device(device),
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        offset = checkpoint['epoch']

        logging.info(f'Checkpoint restored from {final_checkpoint}')
        
        clean_history(
            history_path=history_path,
            epoch_threshold=offset,
        )

    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    for epoch in range(offset, epochs):

        epoch_start_time = time.time()
        model.train()

        counter = 0
        running_loss = 0
        total_samples = 0
        running_corrects = 0

        for batch in tqdm(train_loader):
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            counter += 1

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_str, enabled=use_amp):
                outputs = model(images)
                loss_train = criterion(outputs, labels)

            running_loss += loss_train.item()

            preds = outputs.argmax(1)

            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            scaler.scale(loss_train).backward()
            scaler.step(optimizer)
            scaler.update()  

        train_loss = running_loss / counter
        train_acc = running_corrects / total_samples

        # Validation
        model.eval()
        counter = 0
        running_loss = 0
        total_samples = 0
        running_corrects = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch[0], batch[1]
                images, labels = images.to(device), labels.to(device)
                counter += 1

                outputs = model(images)

                running_loss += criterion(outputs, labels).item()

                preds = outputs.argmax(1)

                running_corrects += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

        val_loss = running_loss / counter
        val_acc = running_corrects / total_samples

        # Changing step sizes
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Epoch {epoch}: Learning rate reduced from {current_lr} to {new_lr}")

        # Saving checkpoints
        if val_loss < (best_loss - threshold):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth')

            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break
        
        # Store loss
        write_to_csv(history_path, epoch, train_loss, train_acc, val_loss, val_acc)

        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(train_loss),
            "acc_train: {:.4f}".format(train_acc),
            "loss_val: {:.4f}".format(val_loss),
            "acc_val: {:.4f}".format(val_acc),
            "time: {:.4f}s".format(time.time() - epoch_start_time),
        )

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))


def test_model(
    model, 
    test_loader,
    checkpoint_dir,
    criterion,
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model = torch.nn.DataParallel(model)

    checkpoints = [x for x in os.listdir(checkpoint_dir) if x.endswith('.pth')]
    checkpoints.sort()

    model.load_state_dict(torch.load(checkpoints[-1], map_location=device))
    model.eval()

    counter = 0
    running_loss = 0
    total_samples = 0
    running_corrects = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            counter += 1

            outputs = model(images)

            preds = outputs.argmax(1)
            label_1_dim = labels.argmax(1)

            running_corrects += torch.sum(preds == label_1_dim).item()
            total_samples += labels.size(0)

            running_loss += criterion(outputs, labels)

    test_loss = running_loss / counter
    test_acc = running_corrects / total_samples

    print("Test set results:", "loss= {:.4f}".format(test_loss), "accuracy= {:.4f}".format(test_acc))
