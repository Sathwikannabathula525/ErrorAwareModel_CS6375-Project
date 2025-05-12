import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os
import argparse
import hashlib
import urllib.request
import zipfile
import shutil
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def get_num_workers():
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except:
        return 2

class Config:
    batch_size = 64
    epochs = 30
    initial_lr = 0.001
    error_prob = 0.3
    error_magnitude = 0.05
    grad_accum_steps = 2
    critical_layers = ['layer3', 'layer4']
    error_mode = "noise"
    log_dir = "logs"

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

def download_tiny_imagenet():
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"
    data_dir = "tiny-imagenet-200"

    if os.path.exists(data_dir) and os.path.isdir(os.path.join(data_dir, "train")):
        print("Tiny ImageNet already exists.")
        return

    print("\u2b07Downloading Tiny ImageNet...")
    with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
        total = int(response.headers.get('content-length', 0))
        with tqdm(total=total, unit='B', unit_scale=True, desc=zip_path) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

    print(" Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

    nested_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(nested_dir):
        for f in os.listdir(nested_dir):
            shutil.move(os.path.join(nested_dir, f), data_dir)
        shutil.rmtree(nested_dir)

    os.remove(zip_path)
    print("Tiny ImageNet ready.")

def create_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 200))
    return model

class CompilerErrorSimulator:
    def __init__(self, error_prob=config.error_prob, error_magnitude=config.error_magnitude):
        self.error_prob = error_prob
        self.error_magnitude = error_magnitude

    def inject_errors(self, tensor):
        if self.error_prob <= 0:
            return tensor
        mask = torch.rand_like(tensor) < self.error_prob
        noise = self.error_magnitude * torch.randn_like(tensor)
        return torch.where(mask, tensor + noise, tensor)

class ErrorAwareTrainingWrapper(nn.Module):
    def __init__(self, model, error_simulator):
        super().__init__()
        self.model = model
        self.error_simulator = error_simulator
        self._register_hooks()

    def forward(self, x):
        x = self.error_simulator.inject_errors(x)
        return self.model(x)

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(lambda m, i, o: self.error_simulator.inject_errors(o))

class ErrorAwareLoss(nn.Module):
    def __init__(self, base_loss, error_simulator, lambda_param=0.3):
        super().__init__()
        self.base_loss = base_loss
        self.error_simulator = error_simulator
        self.lambda_param = lambda_param

    def forward(self, input, target):
        base = self.base_loss(input, target)
        with torch.no_grad():
            perturbed = self.error_simulator.inject_errors(input)
            penalty = self.base_loss(perturbed, target) - base
        return base + self.lambda_param * penalty

def evaluate_model(model, loader, criterion):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast(enabled=use_amp):
                out = model(x)
                loss_sum += criterion(out, y).item()
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
    return loss_sum / len(loader), 100 * correct / total

def train_model(model, train_loader, val_loader, error_simulator, num_epochs=config.epochs, lr=config.initial_lr):
    model.to(device)
    wrapped_model = ErrorAwareTrainingWrapper(model, error_simulator).to(device)
    criterion = ErrorAwareLoss(nn.CrossEntropyLoss(label_smoothing=0.1), error_simulator)
    optimizer = optim.Adam(wrapped_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=use_amp)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        error_simulator.error_prob = min(0.4, 0.05 + epoch * 0.03)
        error_simulator.error_magnitude = min(0.08, 0.01 + epoch * 0.005)

        wrapped_model.train()
        running_loss = correct = total = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                out = wrapped_model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        val_loss, val_acc = evaluate_model(wrapped_model, val_loader, criterion)
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(100 * correct / total)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}: Train Acc={history['train_acc'][-1]:.2f}%, Val Acc={val_acc:.2f}% | Time: {time.time()-start:.1f}s")
        scheduler.step()

    torch.save(model.state_dict(), "error_aware_model_weights.pth")
    return wrapped_model.model, history

def load_tiny_imagenet(batch_size=config.batch_size):
    val_dir = "./tiny-imagenet-200/val"
    img_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    map_dir = os.path.join(val_dir, "mapped")

    if not os.path.exists(map_dir) or not any(os.scandir(map_dir)):
        os.makedirs(map_dir, exist_ok=True)
        with open(ann_file, "r") as f:
            for line in f:
                img_file, cls = line.strip().split('\t')[:2]
                cls_dir = os.path.join(map_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(img_dir, img_file)
                dst = os.path.join(cls_dir, img_file)
                if os.path.exists(src):
                    shutil.copy(src, dst)

    norm = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    train_tf = transforms.Compose([
        transforms.Resize(72),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ])
    test_tf = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        norm
    ])
    workers = get_num_workers()
    train_set = datasets.ImageFolder('./tiny-imagenet-200/train', transform=train_tf)
    val_set = datasets.ImageFolder(map_dir, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def test_compiler_robustness(model, loader, simulator):
    _, clean_acc = evaluate_model(model, loader, nn.CrossEntropyLoss())
    wrapped = ErrorAwareTrainingWrapper(model, simulator).to(device)
    _, error_acc = evaluate_model(wrapped, loader, nn.CrossEntropyLoss())
    print(f"Clean: {clean_acc:.2f}%, With Errors: {error_acc:.2f}%, Drop: {clean_acc - error_acc:.2f}%")
    return clean_acc, error_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_train', action='store_true')
    args = parser.parse_args()

    download_tiny_imagenet()
    train_loader, val_loader = load_tiny_imagenet()
    error_sim = CompilerErrorSimulator()

    model = create_model()
    if os.path.exists("error_aware_model_weights.pth") and not args.force_train:
        model.load_state_dict(torch.load("error_aware_model_weights.pth", map_location=device))
        model.to(device)
        print("\nLoaded error-aware model weights from file")
    else:
        model, history = train_model(model, train_loader, val_loader, error_sim)

    clean_acc, error_acc = test_compiler_robustness(model, val_loader, error_sim)

    print("\nTraining standard model for comparison...")
    torch.cuda.empty_cache()
    standard_model = create_model().to(device)
    if os.path.exists("standard_model_weights.pth") and not args.force_train:
        standard_model.load_state_dict(torch.load("standard_model_weights.pth", map_location=device))
        print("Loaded standard model weights from file")
    else:
        torch.cuda.empty_cache()
        opt = optim.Adam(standard_model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=use_amp)
        for epoch in range(config.epochs):
            start = time.time()
            standard_model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with autocast(enabled=use_amp):
                    loss = loss_fn(standard_model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            print(f"[Standard] Epoch {epoch+1} finished in {time.time()-start:.1f}s")
        torch.save(standard_model.state_dict(), "standard_model_weights.pth")
    print("\nStandard Model Performance:")
    std_clean, std_err = test_compiler_robustness(standard_model, val_loader, error_sim)

    print(f"\nComparison Results:\nError-Aware | Clean: {clean_acc:.2f}%, With Errors: {error_acc:.2f}%")
    print(f"Standard   | Clean: {std_clean:.2f}%, With Errors: {std_err:.2f}%")

if __name__ == "__main__":
    main()
