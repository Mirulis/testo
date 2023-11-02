import torch

from torch import nn, optim
from torch.optim import lr_scheduler

import dataset
import net
import train

data_path = "wineQuality/data/"
data_name = ["winequality-red.csv", "winequality-white.csv"]

if __name__ == "__main__":
    learning_rate = 1e-3
    batch_size = 400
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, test_dataset = dataset.WineDataset.from_csv(
        data_path, data_name[0], 0.25, device
    )

    train_loader = train_dataset.to_dataloader(
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = test_dataset.to_dataloader(
        batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = net.LeNet(11, 32, 64, 64, 32, 6).to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    model = train.early_stopping(
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        patience=100,
        max_epoch=145270,
    )

    acc = train.test(test_loader, model)
    print(f"Accuracy on test set: {acc}%")
