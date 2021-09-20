from torch import nn
from torch.utils.data.dataloader import DataLoader
from model import Net
from dataset import CaptchaDataset
from torch.utils.data import random_split
import torch

train_test_split = 0.9
dataset_path = "C:\\Users\\xyqye\\Files\\tech\\ua-scu-captcha"
lr = 1e-3
batch_size = 256
epochs = 10


def train(model):
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = CaptchaDataset(dataset_path)
    dataset_size = len(dataset)
    train_set_size = int(dataset_size * train_test_split)

    train_set, test_set = random_split(
        dataset, (train_set_size, dataset_size - train_set_size)
    )

    train_set_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_set_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, (x, label) in enumerate(train_set_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            y1, y2, y3, y4, y5, y6 = model(x)
            loss1, loss2, loss3, loss4, loss5, loss6 = (
                criterion(y1, label[:, 0]),
                criterion(y2, label[:, 1]),
                criterion(y3, label[:, 2]),
                criterion(y4, label[:, 3]),
                criterion(y5, label[:, 4]),
                criterion(y6, label[:, 5]),
            )
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            loss.backward()
            optimizer.step()

            print(f"i={i}/{int(train_set_size/batch_size)}, loss={loss:.4}", end="\r")

        print(f"\nepoch={epoch} summary:")
        test_accurate_count = test_accuracy(model, test_set_loader)
        print(
            f"test set accuracy: {test_accurate_count / len(test_set):.4} ({test_accurate_count}/{len(test_set)})"
        )


def test_accuracy(model, data_loader) -> int:
    accurate_count = 0
    for i, (x, label) in enumerate(data_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        y = torch.cat(tuple(torch.argmax(i, dim=1)[:, None] for i in model(x)), 1)
        for j in label == y:
            if torch.all(j):
                accurate_count += 1

        print(f"testing, i={i}", end="\r")

    return accurate_count


if __name__ == "__main__":
    model = Net()
    train(model)
    torch.save(model.state_dict(), "model.pth")
