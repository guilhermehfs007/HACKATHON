"""
Pre-processamento: CLAHE (normalizacao de contraste) + Resize
================================================================
Requer: pip install opencv-python
Ajuste as variaveis de CONFIG e rode: python preprocess_demo.py
"""

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------- CONFIG: AJUSTE AQUI -------------------
IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/images"
OUTPUT_DIR = "eda_output"
NOVO_TAMANHO = (1024, 512)   # (largura, altura) - multiplo de 32
CLIP_LIMIT = 2.0              # limite de contraste do CLAHE (2.0-4.0 e comum)
TILE_GRID_SIZE = (8, 8)       # tamanho dos blocos locais do CLAHE
# -------------------------------------------------------------


def aplicar_clahe(img_gray: np.ndarray) -> np.ndarray:
    """Aplica CLAHE numa imagem em escala de cinza (0-255, uint8)."""
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    return clahe.apply(img_gray)


def redimensionar(img: np.ndarray, tamanho) -> np.ndarray:
    """Redimensiona preservando qualidade (interpolacao de area, boa p/ reducao)."""
    return cv2.resize(img, tamanho, interpolation=cv2.INTER_AREA)


def processar_uma_imagem(caminho_imagem: str):
    """Retorna original, apos CLAHE, e apos CLAHE+resize -- para comparacao."""
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao consegui abrir: {caminho_imagem}")

    img_clahe = aplicar_clahe(img)
    img_final = redimensionar(img_clahe, NOVO_TAMANHO)
    return img, img_clahe, img_final


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pega a primeira imagem da pasta so para demonstracao
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    primeira_imagem = next(f for f in sorted(Path(IMAGES_DIR).iterdir()) if f.suffix.lower() in exts)
    print(f"Processando exemplo: {primeira_imagem.name}")

    original, apos_clahe, final = processar_uma_imagem(str(primeira_imagem))

    print(f"Tamanho original: {original.shape[::-1]}")
    print(f"Tamanho final (apos resize): {final.shape[::-1]}")
    print(f"Brilho medio ANTES do CLAHE: {original.mean():.1f}  (desvio: {original.std():.1f})")
    print(f"Brilho medio DEPOIS do CLAHE: {apos_clahe.mean():.1f}  (desvio: {apos_clahe.std():.1f})")

    # --------- Comparacao visual ---------
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(apos_clahe, cmap="gray")
    axes[1].set_title(f"Apos CLAHE (clip={CLIP_LIMIT})")
    axes[1].axis("off")

    axes[2].imshow(final, cmap="gray")
    axes[2].set_title(f"Final ({NOVO_TAMANHO[0]}x{NOVO_TAMANHO[1]})")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "preprocess_comparacao.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nComparacao visual salva em: {out_path}")
