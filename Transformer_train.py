import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)

class FCBFormer(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8]):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(embed_dims[2], embed_dims[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bridge
        self.bridge = Bridge(embed_dims[-1], embed_dims[-1]*2)

        # Decoder with transposed convolutions
        self.up1 = nn.ConvTranspose2d(embed_dims[-1]*2, embed_dims[-1], 2, stride=2)
        self.dec1 = ConvBlock(embed_dims[-1]+embed_dims[-1], embed_dims[-1])

        self.up2 = nn.ConvTranspose2d(embed_dims[-1], embed_dims[-2], 2, stride=2)
        self.dec2 = ConvBlock(embed_dims[-2]+embed_dims[-2], embed_dims[-2])

        self.up3 = nn.ConvTranspose2d(embed_dims[-2], embed_dims[-3], 2, stride=2)
        self.dec3 = ConvBlock(embed_dims[-3]+embed_dims[-3], embed_dims[-3])

        self.up4 = nn.ConvTranspose2d(embed_dims[-3], embed_dims[-4], 2, stride=2)
        self.dec4 = ConvBlock(embed_dims[-4]+embed_dims[-4], embed_dims[-4])

        # Final prediction
        self.final = nn.Conv2d(embed_dims[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bridge
        b = self.bridge(self.pool4(e4))

        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)

        return self.final(d4)

# Конфигурация
class Config:
    # Пути к данным
    train_img_dir = r"C:\Users\User\PycharmProjects\pythonProject\Imaje recognition\Imaje recognition\FCBFormerOocyte\SegmentationCortex\human\clin1\input"  # Полный путь до train images
    train_mask_dir = r"C:\Users\User\PycharmProjects\pythonProject\Imaje recognition\Imaje recognition\FCBFormerOocyte\SegmentationCortex\human\clin1\mask"  # Полный путь до train masks
    test_img_dir = r"C:\Users\User\PycharmProjects\pythonProject\Imaje recognition\Imaje recognition\FCBFormerOocyte\SegmentationCortex\human\clin1_test\input"  # Полный путь до test images
    test_mask_dir = r"C:\Users\User\PycharmProjects\pythonProject\Imaje recognition\Imaje recognition\FCBFormerOocyte\SegmentationCortex\human\clin1_test\mask"  # Полный путь до test masks

    # Параметры модели
    num_classes = 1  # Для бинарной сегментации
    in_channels = 3
    lr = 1e-4
    batch_size = 30
    num_epochs = 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Кастомный Dataset
# Модифицируем класс Dataset
class OocyteDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Фильтрация только изображений
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
        self.images = self._filter_files(os.listdir(img_dir), img_dir)
        self.masks = self._filter_files(os.listdir(mask_dir), mask_dir)

        # Проверка соответствия имен
        self._validate_filenames()

    def _filter_files(self, files, dir_path):
        """Фильтрует файлы по расширениям изображений"""
        return sorted([
            f for f in files
            if os.path.splitext(f)[1].lower() in self.image_extensions
               and os.path.isfile(os.path.join(dir_path, f))
        ])

    def _validate_filenames(self):
        """Проверяет соответствие имен изображений и масок"""
        img_names = [os.path.splitext(f)[0] for f in self.images]
        mask_names = [os.path.splitext(f)[0] for f in self.masks]

        if img_names != mask_names:
            mismatched = list(set(img_names) ^ set(mask_names))
            raise RuntimeError(
                f"Mismatched files: {mismatched[:3]}..."
                f"\nTotal images: {len(img_names)}, masks: {len(mask_names)}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            # Загрузка с обработкой ошибок
            with Image.open(img_path) as img:
                image = np.array(img.convert("RGB"))

            with Image.open(mask_path) as mask:
                mask = np.array(mask.convert("L"), dtype=np.float32)

            mask = mask / 255.0

            if self.transform:
                augmented = self.transform(image=image, mask=mask, is_check_shapes=False)
                image = augmented['image']
                mask = augmented['mask']

            return image, mask.unsqueeze(0)

        except Exception as e:
            print(f"\nERROR loading {img_name} | {mask_name}: {str(e)}")
            # Возвращаем нулевой тензор чтобы не ломать батч
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)
# Модифицируем аугментации
train_transform = A.Compose([
    A.Resize(256, 256, always_apply=True),  # Принудительный ресайз
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)  # Отключаем проверку размеров

val_transform = A.Compose([
    A.Resize(256, 256, always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)
# Инициализация
cfg = Config()
model = FCBFormer(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(cfg.device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

# Даталоадеры
train_dataset = OocyteDataset(cfg.train_img_dir, cfg.train_mask_dir, train_transform)
test_dataset = OocyteDataset(cfg.test_img_dir, cfg.test_mask_dir, val_transform)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

# Цикл обучения
for epoch in range(cfg.num_epochs):
    model.train()
    total_loss = 0

    # Прогресс-бар для обучения
    train_iter = tqdm(train_loader,
                      desc=f'Epoch {epoch+1}/{cfg.num_epochs} [Train]',
                      bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for images, masks in train_iter:
        images = images.to(cfg.device)
        masks = masks.to(cfg.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_iter.set_postfix({'Loss': f'{loss.item():.4f}'})

    # Валидация с прогресс-баром
    model.eval()
    test_loss = 0

    test_iter = tqdm(test_loader,
                     desc=f'Epoch {epoch+1}/{cfg.num_epochs} [Val]',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for images, masks in test_iter:
            images = images.to(cfg.device)
            masks = masks.to(cfg.device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            test_iter.set_postfix({'Val Loss': f'{loss.item():.4f}'})

    # Вывод статистики
    print(f"\nEpoch {epoch+1} Summary: "
          f"Train Loss: {total_loss/len(train_loader):.4f} | "
          f"Test Loss: {test_loss/len(test_loader):.4f}\n")


# Сохранение модели
torch.save(model.state_dict(), "fcbformer_oocyte.pth")

# Визуализация результатов
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    image, mask = test_dataset[0]
    pred = model(image.unsqueeze(0).to(cfg.device))
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8)

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.title("Input")
plt.imshow(image.permute(1,2,0))
plt.subplot(132)
plt.title("True Mask")
plt.imshow(mask.squeeze(), cmap='gray')
plt.subplot(133)
plt.title("Prediction")
plt.imshow(pred, cmap='gray')
plt.show()
