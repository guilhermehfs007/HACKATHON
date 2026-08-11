import os
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/images"
MASKS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/masks"
OUTPUT_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/eda_output"
N_VARIACOES = 5   # quantas versoes augmentadas gerar para o exemplo
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- Pipeline de augmentation ---------
# Cada transformacao tem uma probabilidade (p) de ser aplicada -- isso gera
# combinacoes diferentes a cada chamada, mesmo pra mesma imagem de entrada.
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-12, 12), translate_percent=(0.0, 0.05), scale=(0.95, 1.05), p=0.7),
    A.ElasticTransform(alpha=40, sigma=6, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
])

# --------- Pega uma imagem de exemplo ---------
imagens = sorted(Path(IMAGES_DIR).glob("*.png"))
img_path = imagens[0]
nome_base = img_path.stem
mask_path = Path(MASKS_DIR) / f"{nome_base}_mask.png"

img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

print(f"Exemplo usado: {nome_base}")

# --------- Gera variacoes e monta grade de comparacao ---------
fig, axes = plt.subplots(2, N_VARIACOES + 1, figsize=(4 * (N_VARIACOES + 1), 8))

axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")
axes[1, 0].imshow(mask, cmap="gray")
axes[1, 0].set_title("Mascara original")
axes[1, 0].axis("off")

for i in range(N_VARIACOES):
    aug = transform(image=img, mask=mask)
    img_aug, mask_aug = aug["image"], aug["mask"]

    axes[0, i + 1].imshow(img_aug, cmap="gray")
    axes[0, i + 1].set_title(f"Augmentada #{i+1}")
    axes[0, i + 1].axis("off")

    axes[1, i + 1].imshow(mask_aug, cmap="gray")
    axes[1, i + 1].axis("off")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "augmentation_demo.png")
plt.savefig(out_path, dpi=120)
print(f"Comparacao salva em: {out_path}")
