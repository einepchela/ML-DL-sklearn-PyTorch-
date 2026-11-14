import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

image_path = './'

def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    acc_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    acc_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch.float()).float()
            acc_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        acc_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float()
                acc_hist_valid[epoch] += is_correct.sum()
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            acc_hist_valid[epoch] /= len(valid_dl.dataset)

            print(f'Точность после эпохи {epoch + 1}: '
                  f'{acc_hist_train[epoch]:.4f} Точность при валидации: '
                  f'{acc_hist_valid[epoch]:.4f}')

    return loss_hist_train, loss_hist_valid, acc_hist_train, acc_hist_valid


celeba_train_dataset = torchvision.datasets.CelebA(
    image_path, split='train', target_type='attr', download=False
)
celeba_valid_dataset = torchvision.datasets.CelebA(
    image_path, split='valid', target_type='attr', download=False
)
celeba_test_dataset = torchvision.datasets.CelebA(
    image_path, split='test', target_type='attr', download=False
)

print('Обучение: ', len(celeba_train_dataset))
print('Валидация: ', len(celeba_valid_dataset))
print('Тестирование: ', len(celeba_test_dataset))

get_smile = lambda attr: attr[31]

transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])
transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])

celeba_train_dataset = torchvision.datasets.CelebA(
    image_path, split='train', target_type='attr', download=False,
    transform=transform_train, target_transform=get_smile
)
torch.manual_seed(1)
data_loader = DataLoader(celeba_train_dataset, batch_size=2)

celeba_valid_dataset = torchvision.datasets.CelebA(
    image_path, split='valid', target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)
celeba_test_dataset = torchvision.datasets.CelebA(
    image_path, split='test', target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)

celeba_train_dataset = Subset(celeba_train_dataset, torch.arange(16000))
celeba_valid_dataset = Subset(celeba_valid_dataset, torch.arange(1000))

batch_size = 32
torch.manual_seed(1)
train_dl = DataLoader(celeba_train_dataset, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(celeba_valid_dataset, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(celeba_test_dataset, batch_size=batch_size, shuffle=False)

model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout1', nn.Dropout(p=0.5))
model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout2', nn.Dropout(p=0.5))
model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))
model.add_module('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
model.add_module('relu4', nn.ReLU())
model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
model.add_module('flatten', nn.Flatten())
model.add_module('fc1', nn.Linear(in_features=256, out_features=1))
model.add_module('sigmoid', nn.Sigmoid())

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.manual_seed(1)
num_epochs = 30
hist = train(model, num_epochs, train_dl, valid_dl)

x_arr = np.arange(len(hist[0]))  + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Потери при обучении')
ax.plot(x_arr, hist[1], '--<', label='Потери при валидации')
ax.set_xlabel('Эпоха', size=15)
ax.set_ylabel('Потери', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Точность при обучении')
ax.plot(x_arr, hist[3], '--<', label='Точность при валидации')
ax.legend(fontsize=15)
ax.set_xlabel('Эпоха', size=15)
ax.set_ylabel('Точность', size=15)
plt.show()

accuracy_test = 0
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_dl:
        pred = model(x_batch)[:, 0]
        is_correct = ((pred >= 0.5).float() == y_batch).float()
        accuracy_test += is_correct.sum()
accuracy_test /= len(test_dl.dataset)
print(f'Точность при тестировании: {accuracy_test:.4f}')

pred = model(x_batch)[:, 0] * 100
fig = plt.figure(figsize=(15, 7))
for j in range(10, 20):
    ax = fig.add_subplot(2, 5, j - 10 + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(x_batch[j].permute(1, 2, 0))
    if y_batch[j] == 1:
        label = 'Улыбка'
    else:
        label = 'Нет улыбки'
    ax.text(0.5, -0.15,
            f'GT: {label:s}\nPr(Smile)={pred[j]:.0f}%',
            size=16,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()