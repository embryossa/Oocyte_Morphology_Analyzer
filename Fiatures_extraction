import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import feature, measure, filters, morphology
from skimage.filters import hessian
from scipy import stats, ndimage, spatial, fft
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import itertools
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import torch.nn as nn

# ##############################
# Модель FCBFormer
# ##############################

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
    def __init__(self, in_channels=3, num_classes=1, embed_dims=[64, 128, 256, 512]):
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

        # Decoder
        self.up1 = nn.ConvTranspose2d(embed_dims[-1]*2, embed_dims[-1], 2, stride=2)
        self.dec1 = ConvBlock(embed_dims[-1]*2, embed_dims[-1])
        self.up2 = nn.ConvTranspose2d(embed_dims[-1], embed_dims[-2], 2, stride=2)
        self.dec2 = ConvBlock(embed_dims[-2]*2, embed_dims[-2])
        self.up3 = nn.ConvTranspose2d(embed_dims[-2], embed_dims[-3], 2, stride=2)
        self.dec3 = ConvBlock(embed_dims[-3]*2, embed_dims[-3])
        self.up4 = nn.ConvTranspose2d(embed_dims[-3], embed_dims[-4], 2, stride=2)
        self.dec4 = ConvBlock(embed_dims[-4]*2, embed_dims[-4])

        self.final = nn.Conv2d(embed_dims[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bridge
        b = self.bridge(self.pool4(e4))

        # Decoder
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

# ##############################
# Анализ морфометрии
# ##############################
class MorphometricsAnalyzer:
    def __init__(self, image, mask):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Shape mismatch: Image {image.shape[:2]}, Mask {mask.shape[:2]}")

        self.image = image
        self.mask = mask
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.masked_gray = cv2.bitwise_and(self.gray, self.gray, mask=self.mask)

    def calculate_moran_index(self, k=5):
        y, x = np.where(self.mask > 0)
        if len(x) == 0:
            return 0  # нет точек в маске

        values = self.masked_gray[self.mask > 0]
        coords = np.column_stack((x, y))
        dist_matrix = spatial.distance_matrix(coords, coords)
        weights = (dist_matrix <= k).astype(float)
        np.fill_diagonal(weights, 0)

        values_norm = values - np.mean(values)
        numerator = np.sum(weights * np.outer(values_norm, values_norm))
        denominator = np.sum(values_norm**2)
        N = len(values)
        W = np.sum(weights)

        return (N/W) * (numerator / denominator) if denominator != 0 else 0

    def loco_efa_analysis(self, contour, n_harmonics=10):
        contour = contour.squeeze().astype(float)
        if contour.ndim != 2 or contour.shape[0] < 2:
            return {'EFA_CumDist': 0, 'EFA_Entropy': 0, 'EFA_MaxMode': 0}

        complex_contour = contour[:, 0] + 1j * contour[:, 1]
        fft_coeffs = fft.fft(complex_contour)
        harmonics = fft_coeffs[:n_harmonics]

        power = np.abs(harmonics)**2
        cum_power = np.cumsum(power) / np.sum(power)
        norm_power = power / np.sum(power)
        entropy = -np.sum(norm_power * np.log(norm_power + 1e-12))

        return {
            'EFA_CumDist': cum_power[-1],
            'EFA_Entropy': entropy,
            'EFA_MaxMode': np.argmax(power)
        }

    def full_analysis(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue

            features = {}

            features.update(self._basic_geometry(cnt))
            features.update(self._curvature_analysis(cnt))
            features.update(self._texture_analysis())
            features['MoranIndex_K5'] = self.calculate_moran_index(5)
            features['MoranIndex_K20'] = self.calculate_moran_index(20)
            features.update(self.loco_efa_analysis(cnt))

            results.append(features)

        return results

    def _basic_geometry(self, cnt):
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (minor, major), angle = ellipse
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        return {
            'Area': area,
            'Perimeter': perimeter,
            'MajorAxis': major,
            'MinorAxis': minor,
            'AspectRatio': major / minor if minor != 0 else 0,
            'Circularity': (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
        }

    def _curvature_analysis(self, cnt):
        cnt = cnt.squeeze().astype(float)
        dx = np.gradient(cnt[:, 0])
        dy = np.gradient(cnt[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2 + 1e-8)**1.5

        return {
            'CurvatureMean': np.mean(curvature),
            'CurvatureStd': np.std(curvature),
            'BendingEnergy': np.mean(curvature**2)
        }

    def _texture_analysis(self):
        glcm = feature.graycomatrix(self.masked_gray, [1], [0], 256, symmetric=True)
        return {
            'GLCM_Contrast': feature.graycoprops(glcm, 'contrast')[0, 0],
            'GLCM_Homogeneity': feature.graycoprops(glcm, 'homogeneity')[0, 0]
        }

# ##############################
# Визуализация и сохранение
# ##############################
class ResultVisualizer:
    @staticmethod
    def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.3):
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    @staticmethod
    def save_analysis_results(results, output_dir):
        if not results or not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
            print("[Warning] No valid analysis results to save.")
            return pd.DataFrame()  # Пустой DataFrame

        df = pd.DataFrame(results)
        csv_path = Path(output_dir) / 'full_analysis.csv'
        df.to_csv(csv_path, index=False)
        return df

    @classmethod
    def visualize_and_save(cls, results, output_dir, image, mask):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        overlay = cls.overlay_mask(image, mask)
        cv2.imwrite(str(Path(output_dir) / 'mask_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(Path(output_dir) / 'binary_mask.png'), mask)

        df = cls.save_analysis_results(results, output_dir)
        if df.empty:
            print("[Info] Skipping visualization: no data.")
            return

        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.scatter(df['Area'], df['Perimeter'], alpha=0.6)
        plt.title('Area vs Perimeter')
        plt.xlabel('Area (px²)')
        plt.ylabel('Perimeter (px)')

        plt.subplot(222)
        plt.hist(df['Circularity'], bins=20, color='skyblue')
        plt.title('Circularity Distribution')
        plt.xlabel('Circularity')

        plt.subplot(223)
        plt.scatter(df['EFA_CumDist'], df['EFA_Entropy'], c=df['EFA_MaxMode'], cmap='viridis')
        plt.title('EFA Analysis')
        plt.colorbar(label='Dominant Mode')

        plt.subplot(224)
        plt.boxplot([df['MoranIndex_K5'], df['MoranIndex_K20']], labels=['K5', 'K20'])
        plt.title('Moran Index Comparison')
        plt.ylabel('Index Value')

        plt.tight_layout()
        plt.savefig(str(Path(output_dir) / 'advanced_analysis.png'))
        plt.close()


# ##############################
# Основной пайплайн
# ##############################

class OocyteAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size
        image_np = np.array(image)

        transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        transformed = transform(image=image_np)
        return transformed['image'].unsqueeze(0), orig_size, image_np

    def postprocess_mask(self, mask_tensor, target_size):
        mask = (torch.sigmoid(mask_tensor).squeeze().cpu().numpy() > self.cfg.threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)

    def analyze(self, image_path):
        tensor, orig_size, orig_image = self.preprocess(image_path)

        model = FCBFormer().to(self.device)
        model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.device))
        model.eval()

        with torch.no_grad():
            output_tensor = model(tensor.to(self.device))

        mask = self.postprocess_mask(output_tensor, orig_size)

        if orig_image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, orig_image.shape[:2][::-1])

        analyzer = MorphometricsAnalyzer(orig_image, mask)
        results = analyzer.full_analysis()

        ResultVisualizer.visualize_and_save(
            results,
            Path(self.cfg.output_dir) / Path(image_path).stem,
            orig_image,
            mask
        )
        return results

# ##############################
# Конфигурация и запуск
# ##############################

class Config:
    def __init__(self):
        self.model_path = "fcbformer_oocyte.pth"
        self.input_path = "C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/FCBFormerOocyte/SegmentationCortex/human/clin1_test/input/ovo_47_t151.png"
        self.output_dir = "results"
        self.threshold = 0.5

if __name__ == "__main__":
    cfg = Config()
    analyzer = OocyteAnalyzer(cfg)
    results = analyzer.analyze(cfg.input_path)

    print("\nПример результатов:")
    if results:
        print(pd.DataFrame(results).head())
    else:
        print("Результаты отсутствуют.")

