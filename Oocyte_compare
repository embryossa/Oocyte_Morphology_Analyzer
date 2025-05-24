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
import glob
import os
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

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

    def calculate_moran_index(self, k=5, max_points=5000):
        """Вычисляем индекс Морана с ограничением на количество точек для предотвращения переполнения памяти"""
        y, x = np.where(self.mask > 0)
        if len(x) == 0:
            return 0  # нет точек в маске

        # Ограничиваем количество точек для предотвращения переполнения памяти
        if len(x) > max_points:
            # Случайно выбираем подмножество точек
            indices = np.random.choice(len(x), max_points, replace=False)
            x = x[indices]
            y = y[indices]

        values = self.masked_gray[y, x]  # Исправлено: сначала y, потом x
        coords = np.column_stack((x, y))

        try:
            dist_matrix = spatial.distance_matrix(coords, coords)
            weights = (dist_matrix <= k).astype(float)
            np.fill_diagonal(weights, 0)

            values_norm = values - np.mean(values)
            numerator = np.sum(weights * np.outer(values_norm, values_norm))
            denominator = np.sum(values_norm**2)
            N = len(values)
            W = np.sum(weights)

            return (N/W) * (numerator / denominator) if denominator != 0 and W != 0 else 0
        except MemoryError:
            print(f"Memory error in Moran index calculation, too many points: {len(x)}")
            return 0
        except Exception as e:
            print(f"Error in Moran index calculation: {str(e)}")
            return 0

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
        """Полный анализ с обработкой ошибок памяти"""
        try:
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            results = []

            for cnt in contours:
                if cv2.contourArea(cnt) < 100:
                    continue

                try:
                    features = {}

                    features.update(self._basic_geometry(cnt))
                    features.update(self._curvature_analysis(cnt))
                    features.update(self._texture_analysis())

                    # Вычисляем индексы Морана с обработкой ошибок
                    try:
                        features['MoranIndex_K5'] = self.calculate_moran_index(5)
                        features['MoranIndex_K20'] = self.calculate_moran_index(20)
                    except Exception as e:
                        print(f"Warning: Moran index calculation failed: {str(e)}")
                        features['MoranIndex_K5'] = 0
                        features['MoranIndex_K20'] = 0

                    try:
                        features.update(self.loco_efa_analysis(cnt))
                    except Exception as e:
                        print(f"Warning: EFA analysis failed: {str(e)}")
                        features.update({'EFA_CumDist': 0, 'EFA_Entropy': 0, 'EFA_MaxMode': 0})

                    results.append(features)

                except Exception as e:
                    print(f"Warning: Failed to analyze contour: {str(e)}")
                    continue

            return results

        except Exception as e:
            print(f"Error in full analysis: {str(e)}")
            return []

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
        """Анализ текстуры с обработкой ошибок"""
        try:
            # Убеждаемся, что изображение имеет правильный диапазон значений
            masked_gray_norm = self.masked_gray.copy()
            if masked_gray_norm.max() > 255:
                masked_gray_norm = (masked_gray_norm / masked_gray_norm.max() * 255).astype(np.uint8)

            # Ограничиваем количество уровней серого для GLCM
            levels = min(256, len(np.unique(masked_gray_norm)))
            if levels < 2:
                return {'GLCM_Contrast': 0, 'GLCM_Homogeneity': 0}

            glcm = feature.graycomatrix(masked_gray_norm, [1], [0], levels, symmetric=True)

            return {
                'GLCM_Contrast': feature.graycoprops(glcm, 'contrast')[0, 0],
                'GLCM_Homogeneity': feature.graycoprops(glcm, 'homogeneity')[0, 0]
            }
        except Exception as e:
            print(f"Warning: Texture analysis failed: {str(e)}")
            return {'GLCM_Contrast': 0, 'GLCM_Homogeneity': 0}

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
    def save_analysis_results(results, output_path, class_label):
        if not results or not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
            print(f"[Warning] No valid analysis results to save for {class_label}.")
            return pd.DataFrame()

        # Добавляем информацию о классе к каждому результату
        for result in results:
            result['class'] = class_label

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} results to {output_path}")
        return df

    @staticmethod
    def create_comparison_plots(normal_df, abnormal_df, output_dir):
        """Создает сравнительные графики между нормальными и аномальными ооцитами"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Список метрик для сравнения
        metrics = ['Area', 'Perimeter', 'AspectRatio', 'Circularity',
                   'CurvatureMean', 'BendingEnergy', 'MoranIndex_K5',
                   'EFA_Entropy', 'GLCM_Contrast']

        # Создаем большой график со всеми сравнениями
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break

            ax = axes[i]

            # Проверяем наличие метрики в обеих группах
            if metric in normal_df.columns and metric in abnormal_df.columns:
                normal_values = normal_df[metric].dropna()
                abnormal_values = abnormal_df[metric].dropna()

                # Создаем гистограмму
                ax.hist(normal_values, alpha=0.6, label='Normal', bins=20, color='green')
                ax.hist(abnormal_values, alpha=0.6, label='Abnormal', bins=20, color='red')
                ax.set_title(f'{metric} Distribution')
                ax.set_xlabel(metric)
                ax.set_ylabel('Frequency')
                ax.legend()

                # Статистический тест
                try:
                    stat, p_value = ttest_ind(normal_values, abnormal_values)
                    ax.text(0.02, 0.98, f'p-value: {p_value:.4f}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                except:
                    pass
            else:
                ax.text(0.5, 0.5, f'No data for {metric}',
                        transform=ax.transAxes, ha='center', va='center')

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'morphology_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Создаем боксплот для основных метрик
        plt.figure(figsize=(15, 10))
        combined_df = pd.concat([normal_df, abnormal_df], ignore_index=True)

        key_metrics = ['Area', 'Circularity', 'AspectRatio', 'CurvatureMean']

        for i, metric in enumerate(key_metrics, 1):
            plt.subplot(2, 2, i)
            if metric in combined_df.columns:
                sns.boxplot(data=combined_df, x='class', y=metric)
                plt.title(f'{metric} by Class')

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def generate_statistical_report(normal_df, abnormal_df, output_dir):
        """Генерирует статистический отчет сравнения"""
        report_path = Path(output_dir) / 'statistical_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("STATISTICAL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Normal oocytes: {len(normal_df)} samples\n")
            f.write(f"Abnormal oocytes: {len(abnormal_df)} samples\n\n")

            # Описательная статистика
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("-" * 30 + "\n\n")

            metrics = ['Area', 'Perimeter', 'AspectRatio', 'Circularity',
                       'CurvatureMean', 'BendingEnergy', 'MoranIndex_K5']

            for metric in metrics:
                if metric in normal_df.columns and metric in abnormal_df.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  Normal   - Mean: {normal_df[metric].mean():.4f}, Std: {normal_df[metric].std():.4f}\n")
                    f.write(f"  Abnormal - Mean: {abnormal_df[metric].mean():.4f}, Std: {abnormal_df[metric].std():.4f}\n")

                    # Статистические тесты
                    try:
                        normal_vals = normal_df[metric].dropna()
                        abnormal_vals = abnormal_df[metric].dropna()

                        # t-test
                        t_stat, t_pval = ttest_ind(normal_vals, abnormal_vals)
                        f.write(f"  t-test p-value: {t_pval:.6f}\n")

                        # Mann-Whitney U test
                        u_stat, u_pval = mannwhitneyu(normal_vals, abnormal_vals, alternative='two-sided')
                        f.write(f"  Mann-Whitney U p-value: {u_pval:.6f}\n")

                        # Эффект размера (Cohen's d)
                        pooled_std = np.sqrt(((len(normal_vals)-1)*normal_vals.var() +
                                              (len(abnormal_vals)-1)*abnormal_vals.var()) /
                                             (len(normal_vals) + len(abnormal_vals) - 2))
                        cohens_d = (normal_vals.mean() - abnormal_vals.mean()) / pooled_std
                        f.write(f"  Cohen's d: {cohens_d:.4f}\n")

                    except Exception as e:
                        f.write(f"  Statistical test failed: {str(e)}\n")

                    f.write("\n")

        print(f"Statistical report saved to {report_path}")

# ##############################
# Основной пайплайн для сравнения
# ##############################

class ComparativeOocyteAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_model(self):
        """Загружаем модель один раз"""
        if self.model is None:
            self.model = FCBFormer().to(self.device)
            self.model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully")

    def preprocess(self, image_path):
        """Предобработка с проверкой размера изображения"""
        try:
            image = Image.open(image_path).convert("RGB")
            orig_size = image.size

            # Проверяем размер изображения
            width, height = orig_size
            max_dimension = 4096  # Максимальный размер для обработки

            if width > max_dimension or height > max_dimension:
                print(f"Warning: Large image {width}x{height}, resizing for processing")
                # Пропорционально уменьшаем изображение
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * max_dimension / width)
                else:
                    new_height = max_dimension
                    new_width = int(width * max_dimension / height)

                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Resized to: {new_width}x{new_height}")

            image_np = np.array(image)

            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            transformed = transform(image=image_np)
            return transformed['image'].unsqueeze(0), orig_size, image_np

        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}")

    def postprocess_mask(self, mask_tensor, target_size):
        mask = (torch.sigmoid(mask_tensor).squeeze().cpu().numpy() > self.cfg.threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)

    def analyze_single_image(self, image_path):
        """Анализируем одно изображение"""
        try:
            # Проверяем, что файл существует и является файлом
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")

            # Проверяем размер файла
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"Empty file: {image_path}")

            # Пытаемся открыть изображение
            try:
                tensor, orig_size, orig_image = self.preprocess(image_path)
            except Exception as e:
                raise ValueError(f"Cannot preprocess image: {str(e)}")

            # Проверяем размеры
            if orig_image.size == 0:
                raise ValueError(f"Invalid image dimensions")

            with torch.no_grad():
                output_tensor = self.model(tensor.to(self.device))

            mask = self.postprocess_mask(output_tensor, orig_size)

            if orig_image.shape[:2] != mask.shape:
                mask = cv2.resize(mask, orig_image.shape[:2][::-1])

            analyzer = MorphometricsAnalyzer(orig_image, mask)
            results = analyzer.full_analysis()

            # Добавляем имя файла к каждому результату
            for result in results:
                result['filename'] = Path(image_path).name

            return results

        except Exception as e:
            print(f"Detailed error for {os.path.basename(image_path)}: {str(e)}")
            print(f"Full path: {image_path}")
            # Пробуем получить дополнительную информацию о файле
            try:
                if os.path.exists(image_path):
                    print(f"File size: {os.path.getsize(image_path)} bytes")
                    print(f"File extension: {os.path.splitext(image_path)[1]}")
            except:
                pass
            return []

    def analyze_folder(self, folder_path, class_label):
        """Анализируем все изображения в папке"""
        print(f"Analyzing {class_label} images from: {folder_path}")

        # Проверяем, существует ли папка
        if not os.path.exists(folder_path):
            print(f"Error: Folder does not exist: {folder_path}")
            return []

        # Поддерживаемые форматы изображений (только в основной папке, не в подпапках)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []

        for ext in image_extensions:
            # Используем glob только для основной папки, не рекурсивно
            paths_lower = glob.glob(os.path.join(folder_path, ext))
            paths_upper = glob.glob(os.path.join(folder_path, ext.upper()))
            image_paths.extend(paths_lower)
            image_paths.extend(paths_upper)

        # Удаляем дубликаты и сортируем
        image_paths = sorted(list(set(image_paths)))

        # Фильтруем только файлы (не папки)
        image_paths = [p for p in image_paths if os.path.isfile(p)]

        print(f"Files found by extension:")
        for ext in image_extensions:
            count_lower = len(glob.glob(os.path.join(folder_path, ext)))
            count_upper = len(glob.glob(os.path.join(folder_path, ext.upper())))
            if count_lower > 0 or count_upper > 0:
                print(f"  {ext}: {count_lower + count_upper} files")

        if not image_paths:
            print(f"No images found in {folder_path}")
            # Покажем что вообще есть в папке
            all_files = os.listdir(folder_path)
            print(f"All files in directory: {all_files[:10]}{'...' if len(all_files) > 10 else ''}")
            return []

        print(f"Total unique image files found: {len(image_paths)}")
        print(f"First few files: {[os.path.basename(p) for p in image_paths[:5]]}")

        all_results = []
        failed_files = []

        # Обрабатываем изображения с прогресс-баром
        for image_path in tqdm(image_paths, desc=f"Processing {class_label}"):
            try:
                results = self.analyze_single_image(image_path)
                all_results.extend(results)
            except Exception as e:
                failed_files.append((image_path, str(e)))
                print(f"\nFailed to process {os.path.basename(image_path)}: {str(e)}")
                continue

        print(f"Completed analysis of {class_label}:")
        print(f"  Successfully processed: {len(image_paths) - len(failed_files)} images")
        print(f"  Failed: {len(failed_files)} images")
        print(f"  Total objects detected: {len(all_results)}")

        if failed_files:
            print("Failed files:")
            for file_path, error in failed_files[:5]:  # Показываем первые 5 ошибок
                print(f"  {os.path.basename(file_path)}: {error}")

        return all_results

    def run_comparative_analysis(self):
        """Запускаем полный сравнительный анализ"""
        print("Starting comparative oocyte analysis...")

        # Загружаем модель
        self.load_model()

        # Создаем выходную директорию
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Анализируем нормальные ооциты
        normal_results = self.analyze_folder(self.cfg.normal_path, "normal")

        # Анализируем аномальные ооциты
        abnormal_results = self.analyze_folder(self.cfg.abnormal_path, "abnormal")

        # Сохраняем результаты в CSV файлы
        normal_df = ResultVisualizer.save_analysis_results(
            normal_results,
            output_dir / 'normal_morphometrics.csv',
            'normal'
        )

        abnormal_df = ResultVisualizer.save_analysis_results(
            abnormal_results,
            output_dir / 'abnormal_morphometrics.csv',
            'abnormal'
        )

        # Создаем сравнительные графики и отчеты
        if not normal_df.empty and not abnormal_df.empty:
            print("Generating comparison visualizations...")
            ResultVisualizer.create_comparison_plots(normal_df, abnormal_df, output_dir)
            ResultVisualizer.generate_statistical_report(normal_df, abnormal_df, output_dir)

            # Сохраняем объединенный датасет
            combined_df = pd.concat([normal_df, abnormal_df], ignore_index=True)
            combined_df.to_csv(output_dir / 'combined_morphometrics.csv', index=False)
            print(f"Combined dataset saved: {len(combined_df)} total objects")

        else:
            print("Warning: One or both datasets are empty. Cannot create comparisons.")

        print(f"Analysis complete! Results saved to: {output_dir}")
        return normal_results, abnormal_results

# ##############################
# Конфигурация и запуск
# ##############################

class Config:
    def __init__(self):
        self.model_path = "fcbformer_oocyte.pth"
        self.normal_path = r"C:\Users\User\PycharmProjects\pythonProject\Imaje recognition\Imaje recognition\FCBFormerOocyte\SegmentationCortex\human\normal"
        self.abnormal_path = r"C:\Users\User\PycharmProjects\pythonProject\Imaje recognition\Imaje recognition\FCBFormerOocyte\SegmentationCortex\human\abnormal"
        self.output_dir = "comparative_results"
        self.threshold = 0.5

if __name__ == "__main__":
    cfg = Config()
    analyzer = ComparativeOocyteAnalyzer(cfg)
    normal_results, abnormal_results = analyzer.run_comparative_analysis()

    print(f"\nAnalysis Summary:")
    print(f"Normal oocytes: {len(normal_results)} objects detected")
    print(f"Abnormal oocytes: {len(abnormal_results)} objects detected")
    print(f"Results saved to: {cfg.output_dir}")
