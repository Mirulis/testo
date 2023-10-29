import torch
import torch.nn as nn
from torch import optim

import dataset
import net
import train

data_path = "wineQuality/data/"
data_name = ["winequality-red.csv", "winequality-white.csv"]


if __name__ == "__main__":
    learning_rate = 1e-3
    batch_size = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, test_dataset = dataset.WineDataset.from_csv(
        data_path, data_name[0], 0.25, device
    )

    train_loader = train_dataset.to_dataloader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = test_dataset.to_dataloader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = net.LeNet(11, 22, 22, 6)

    model = train.early_stopping(
        train_loader,
        test_loader,
        model,
        nn.CrossEntropyLoss(reduction="sum"),
        optim.SGD(model.parameters(), lr=learning_rate),
        device,
    )
    train.test(test_loader, model)
