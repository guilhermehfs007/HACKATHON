import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/images"
MASKS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/masks"
OUTPUT_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/eda_output"
N_EXEMPLOS = 6   # quantas amostras mostrar na grade
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

imagens = sorted(Path(IMAGES_DIR).glob("*.png"))[:N_EXEMPLOS]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, img_path in enumerate(imagens):
    nome_base = img_path.stem
    mask_path = Path(MASKS_DIR) / f"{nome_base}_mask.png"

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Converter imagem pra RGB para podermos colorir a mascara em vermelho
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]  # pinta de vermelho onde a mascara diz "dente"

    # Mistura 60% imagem original + 40% overlay vermelho (transparencia)
    resultado = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    axes[i].imshow(resultado)
    axes[i].set_title(nome_base[:20], fontsize=8)
    axes[i].axis("off")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "verificacao_alinhamento.png")
plt.savefig(out_path, dpi=120)
print(f"Verificacao salva em: {out_path}")
