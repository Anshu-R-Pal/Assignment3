

# train.py

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import wandb

def train_model(
    model,
    train_loader,
    test_loader,
    epochs=100,
    batch_size=128,
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    device="cuda"
):
    # Initialize WandB
    wandb.login()
    wandb.init(project="Assignment3_training", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "optimizer": "SGD",
        "scheduler": "StepLR"
    })

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # 🔥 AUTO-PRINT FOR Q1(b)
    print("Training Configuration:")
    print(f"Optimizer: SGD")
    print(f"Learning Rate: {lr}")
    print(f"Momentum: {momentum}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Scheduler: StepLR(step_size=30, gamma=0.1)")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}\n")

    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = loss_sum / total
        train_acc = 100. * correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Evaluate on test set
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss_sum += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

        test_loss = loss_sum / total
        test_acc = 100. * correct / total
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% "
              f"| Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

        # Log to WandB
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch": epoch+1
        })

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list

def plot_curves(train_loss, test_loss, train_acc, test_acc):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_acc, label='Test Acc')
    plt.legend(); plt.title('Accuracy')
    plt.show()


