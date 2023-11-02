from __future__ import annotations


import torch

from tqdm import trange
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from matplotlib import pyplot as plt
from collections.abc import Iterable


def single_train(
    loader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str,
):
    losses = []
    model.train()
    model.to(device)
    for data, labels in loader:
        output = model(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.detach().item())
        optimizer.step()
    scheduler.step()
    return sum(losses) / len(losses)


def early_stopping(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str,
    patience: int = 5,
    max_epoch: int = 10000,
) -> Module:
    counter = 0

    train_losses = []
    eval_losses = []
    best_eval_loss = float("inf")
    process = trange(max_epoch, unit="epoch")
    for epoch in process:
        process.set_description(f"Epoch {epoch + 1}:")
        train_losses.append(
            single_train(
                train_dataloader, model, criterion, optimizer, scheduler, device
            )
        )
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for data, labels in eval_dataloader:
                output = model(data)
                eval_loss += criterion(output, labels).detach().item()
        eval_loss /= len(eval_dataloader)
        eval_losses.append(eval_loss)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    draw_loss_curve(train_loss=train_losses, eval_loss=eval_losses)
    return model


def draw_loss_curve(**keyed_loss: Iterable[float]):
    plt.figure(figsize=(24, 4))
    color_step = 1 / len(keyed_loss)
    for i, (key, losses) in enumerate(keyed_loss.items()):
        # noinspection PyTypeChecker
        plt.plot(losses, label=key, color=(i * color_step, 0.5, 1 - i * color_step))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("current_train_result.png", dpi=2000, bbox_inches="tight")
    plt.show()


def test(loader: DataLoader, model: Module):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in loader:
            output = model(data)
            compare_result = (
                torch.max(labels, dim=1).indices == torch.max(output, dim=1).indices
            )
            # noinspection PyTypeChecker
            correct += torch.sum(compare_result).item()
            total += labels.size(0)
    acc = 100 * correct / total
    return acc
