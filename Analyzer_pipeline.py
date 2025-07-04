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
import lightgbm as lgb

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

        # Проверяем размер маски и при необходимости уменьшаем разрешение для индекса Морана
        self.mask_points = np.sum(self.mask > 0)
        print(f"Количество пикселей в маске: {self.mask_points}")

        if self.mask_points > 50000:
            print("Большая маска detected. Применяем оптимизацию для расчета индекса Морана...")
            # Уменьшаем маску для расчета индекса Морана
            scale_factor = min(1.0, np.sqrt(10000 / self.mask_points))
            new_height = int(self.mask.shape[0] * scale_factor)
            new_width = int(self.mask.shape[1] * scale_factor)

            self.moran_mask = cv2.resize(self.mask, (new_width, new_height),
                                         interpolation=cv2.INTER_NEAREST)
            self.moran_gray = cv2.resize(self.masked_gray, (new_width, new_height),
                                         interpolation=cv2.INTER_LINEAR)
            print(f"Размер маски для индекса Морана уменьшен до: {new_width}x{new_height}")
        else:
            self.moran_mask = self.mask
            self.moran_gray = self.masked_gray

    def calculate_moran_index(self, k=5, max_points=2000):
        # Используем оптимизированную маску для расчета
        y, x = np.where(self.moran_mask > 0)
        if len(x) == 0:
            return 0  # нет точек в маске

        # Дополнительное ограничение количества точек
        if len(x) > max_points:
            # Равномерная выборка точек из маски
            step = len(x) // max_points
            indices = np.arange(0, len(x), step)[:max_points]
            x = x[indices]
            y = y[indices]

        values = self.moran_gray[y, x]  # Используем правильный порядок индексов

        # Удаляем нулевые значения (пиксели вне маски)
        non_zero_mask = values > 0
        if np.sum(non_zero_mask) < 2:
            return 0

        values = values[non_zero_mask]
        x = x[non_zero_mask]
        y = y[non_zero_mask]

        coords = np.column_stack((x, y))

        # Оптимизированное вычисление весов без создания полной матрицы расстояний
        N = len(values)
        values_norm = values - np.mean(values)

        numerator = 0
        W = 0

        # Вычисляем веса и произведения по частям
        for i in range(N):
            # Вычисляем расстояния от точки i до всех остальных
            distances = np.sqrt((coords[i, 0] - coords[:, 0])**2 +
                                (coords[i, 1] - coords[:, 1])**2)

            # Находим соседей в радиусе k
            neighbors = (distances <= k) & (distances > 0)  # исключаем саму точку

            if np.any(neighbors):
                # Добавляем к числителю произведения нормированных значений
                numerator += np.sum(values_norm[i] * values_norm[neighbors])
                W += np.sum(neighbors)

        denominator = np.sum(values_norm**2)

        return (N/W) * (numerator / denominator) if denominator != 0 and W != 0 else 0

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

            try:
                features.update(self._basic_geometry(cnt))
                features.update(self._curvature_analysis(cnt))
                features.update(self._texture_analysis())

                # Безопасное вычисление индекса Морана
                print("Вычисление индекса Морана K=5...")
                features['MoranIndex_K5'] = self.calculate_moran_index(5)
                print("Вычисление индекса Морана K=20...")
                features['MoranIndex_K20'] = self.calculate_moran_index(20)

                features.update(self.loco_efa_analysis(cnt))

                results.append(features)

            except Exception as e:
                print(f"Ошибка при анализе контура: {e}")
                # Возвращаем базовые признаки с нулевыми значениями для проблемных
                features = {
                    'Area': cv2.contourArea(cnt),
                    'Perimeter': cv2.arcLength(cnt, True),
                    'MajorAxis': 0, 'MinorAxis': 0, 'AspectRatio': 1,
                    'Circularity': 0, 'CurvatureMean': 0, 'CurvatureStd': 0,
                    'BendingEnergy': 0, 'GLCM_Contrast': 0, 'GLCM_Homogeneity': 0,
                    'MoranIndex_K5': 0, 'MoranIndex_K20': 0,
                    'EFA_CumDist': 0, 'EFA_Entropy': 0, 'EFA_MaxMode': 0
                }
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
# Объединенный анализатор ооцитов
# ##############################

class UnifiedOocyteAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка модели сегментации
        self.segmentation_model = FCBFormer().to(self.device)
        self.segmentation_model.load_state_dict(
            torch.load(self.cfg.segmentation_model_path, map_location=self.device)
        )
        self.segmentation_model.eval()

        # Загрузка модели классификации
        self.classification_model = lgb.Booster(model_file=self.cfg.classification_model_path)

        # Необходимые колонки для классификации
        self.required_columns = [
            'Area', 'Perimeter', 'MajorAxis', 'MinorAxis', 'AspectRatio', 'Circularity',
            'CurvatureMean', 'CurvatureStd', 'BendingEnergy', 'GLCM_Contrast', 'GLCM_Homogeneity',
            'MoranIndex_K5', 'MoranIndex_K20', 'EFA_CumDist', 'EFA_Entropy', 'EFA_MaxMode'
        ]

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

    def create_colored_overlay(self, image, mask, is_abnormal, alpha=0.3):
        """Создает наложение маски с цветом в зависимости от класса"""
        colored_mask = np.zeros_like(image)
        if is_abnormal:
            color = (255, 0, 0)  # Красный для abnormal
        else:
            color = (0, 255, 0)   # Зеленый для normal

        colored_mask[mask > 0] = color
        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    def analyze(self, image_path):
        print(f"Анализ изображения: {image_path}")

        # 1. Сегментация
        tensor, orig_size, orig_image = self.preprocess(image_path)

        with torch.no_grad():
            output_tensor = self.segmentation_model(tensor.to(self.device))

        mask = self.postprocess_mask(output_tensor, orig_size)

        if orig_image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, orig_image.shape[:2][::-1])

        # 2. Извлечение морфометрических признаков
        analyzer = MorphometricsAnalyzer(orig_image, mask)
        morphometric_results = analyzer.full_analysis()

        if not morphometric_results:
            print("Не удалось извлечь признаки из изображения")
            return None

        # 3. Подготовка данных для классификации
        features_df = pd.DataFrame(morphometric_results)

        # Усреднение признаков, если найдено несколько контуров
        if len(features_df) > 1:
            avg_features = features_df[self.required_columns].mean().to_frame().T
        else:
            avg_features = features_df[self.required_columns]

        # Заполнение пропусков
        avg_features = avg_features.fillna(avg_features.mean())

        # 4. Классификация
        probabilities = self.classification_model.predict(avg_features)
        probability = probabilities[0] if isinstance(probabilities, np.ndarray) else probabilities

        is_abnormal = probability > 0.5
        predicted_class = 'abnormal' if is_abnormal else 'normal'

        # 5. Создание визуализации
        overlay_image = self.create_colored_overlay(orig_image, mask, is_abnormal)

        # 6. Подготовка результатов
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'probability': float(probability),
            'confidence': f"{probability:.3f}" if is_abnormal else f"{1-probability:.3f}",
            'morphometric_features': avg_features.iloc[0].to_dict(),
            'overlay_image': overlay_image,
            'binary_mask': mask,
            'original_image': orig_image
        }

        return result

    def display_results(self, result):
        """Отображение результатов анализа"""
        if result is None:
            print("Нет результатов для отображения")
            return

        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА МОРФОЛОГИИ ООЦИТА")
        print("="*60)

        print(f"Изображение: {Path(result['image_path']).name}")
        print(f"Предсказанный класс: {result['predicted_class'].upper()}")
        print(f"Вероятность аномалии: {result['probability']:.3f}")
        print(f"Уверенность: {result['confidence']}")

        print("\nМОРФОМЕТРИЧЕСКИЕ ХАРАКТЕРИСТИКИ:")
        print("-" * 40)

        features = result['morphometric_features']

        # Основные геометрические параметры
        print("Геометрические параметры:")
        print(f"  Площадь: {features['Area']:.2f} пикс²")
        print(f"  Периметр: {features['Perimeter']:.2f} пикс")
        print(f"  Большая ось: {features['MajorAxis']:.2f} пикс")
        print(f"  Малая ось: {features['MinorAxis']:.2f} пикс")
        print(f"  Соотношение сторон: {features['AspectRatio']:.3f}")
        print(f"  Округлость: {features['Circularity']:.3f}")

        # Параметры кривизны
        print("\nПараметры кривизны:")
        print(f"  Средняя кривизна: {features['CurvatureMean']:.6f}")
        print(f"  Стд. откл. кривизны: {features['CurvatureStd']:.6f}")
        print(f"  Энергия изгиба: {features['BendingEnergy']:.6f}")

        # Текстурные характеристики
        print("\nТекстурные характеристики:")
        print(f"  GLCM Контраст: {features['GLCM_Contrast']:.6f}")
        print(f"  GLCM Однородность: {features['GLCM_Homogeneity']:.6f}")

        # Пространственная автокорреляция
        print("\nПространственная автокорреляция:")
        print(f"  Индекс Морана (K=5): {features['MoranIndex_K5']:.6f}")
        print(f"  Индекс Морана (K=20): {features['MoranIndex_K20']:.6f}")

        # EFA анализ
        print("\nEFA анализ формы:")
        print(f"  EFA Накопленное расстояние: {features['EFA_CumDist']:.6f}")
        print(f"  EFA Энтропия: {features['EFA_Entropy']:.6f}")
        print(f"  EFA Доминирующая мода: {features['EFA_MaxMode']:.0f}")

        # Визуализация
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(result['original_image'])
        plt.title('Исходное изображение')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(result['binary_mask'], cmap='gray')
        plt.title('Бинарная маска')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(result['overlay_image'])
        color_name = "КРАСНЫЙ (abnormal)" if result['predicted_class'] == 'abnormal' else "ЗЕЛЕНЫЙ (normal)"
        plt.title(f'Результат классификации\n{color_name}')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# ##############################
# Конфигурация и запуск
# ##############################

class Config:
    def __init__(self):
        self.segmentation_model_path = "fcbformer_oocyte.pth"
        self.classification_model_path = "oocyte_classifier.txt"
        self.threshold = 0.5

def main():
    """Основная функция для демонстрации работы анализатора"""

    # Пример использования
    cfg = Config()
    analyzer = UnifiedOocyteAnalyzer(cfg)

    # Путь к изображению для анализа
    image_path = "C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/FCBFormerOocyte/SegmentationCortex/human/clin1_test/input/eovo_453_t1.png"

    try:
        # Анализ изображения
        result = analyzer.analyze(image_path)

        # Отображение результатов
        analyzer.display_results(result)

    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
