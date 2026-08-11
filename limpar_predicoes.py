"""
Limpeza pos-predicao -- sem retreinar o modelo
==================================================
O modelo treinou por poucas epocas (restricao de tempo) e a predicao sai com
ruido granulado nas bordas ("sal e pimenta"). Isso nao conserta o problema de
fundo (o modelo ainda nao aprendeu contorno por dente), mas:
  1. Limpa o ruido visual -- fica bem melhor pros slides do pitch.
  2. Costuma recuperar um pouco de Dice, porque parte do ruido conta como
     falso positivo espalhado pela imagem toda.

Tecnica: abertura morfologica (remove pontinhos soltos) + fechamento
(preenche buraquinhos) + remove componentes conectados muito pequenos
(sobras de ruido que a abertura沒 nao pegou sozinha).

Gera uma comparacao de 4 colunas: Original | Predicao bruta | Predicao limpa
| Ground truth -- e um csv com Dice ANTES e DEPOIS da limpeza por imagem.
"""

import csv
from pathlib import Path

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
PROCESSED_IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/images"
PROCESSED_MASKS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/masks"
SPLITS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/splits"
CHECKPOINT_PATH = "/home/akinoriii/VSC/VSCPP/Hackathon/treino_output/melhor_modelo.pt"
OUTPUT_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/treino_output"

ENCODER = "resnet34"
LIMIAR = 0.5

KERNEL_SIZE = 5          # tamanho do elemento estruturante (impar, tipico 3-9)
AREA_MINIMA_PCT = 0.5    # componentes conectados menores que isso (% da area total) somem
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


def dice_score(pred_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    tp = np.logical_and(pred_bin, mask_bin).sum()
    return (2 * tp + 1e-7) / (pred_bin.sum() + mask_bin.sum() + 1e-7)


def limpar_mascara(pred_bin: np.ndarray) -> np.ndarray:
    """Abertura + fechamento + remocao de componentes conectados pequenos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    mask = (pred_bin * 255).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove ruido solto
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # preenche buraquinhos

    # remove componentes conectados muito pequenos (sobras de ruido)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area_total = mask.shape[0] * mask.shape[1]
    area_minima = area_total * (AREA_MINIMA_PCT / 100.0)

    mask_limpa = np.zeros_like(mask)
    for i in range(1, n_labels):  # 0 e o fundo
        if stats[i, cv2.CC_STAT_AREA] >= area_minima:
            mask_limpa[labels == i] = 255

    return (mask_limpa > 0).astype(np.uint8)


def carregar_e_prever(model, stem):
    img = cv2.imread(str(Path(PROCESSED_IMAGES_DIR) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    mask_gt = cv2.imread(str(Path(PROCESSED_MASKS_DIR) / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    mask_gt_bin = (mask_gt > 0).astype(np.uint8)

    tensor = transform(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    pred_bruta = (pred > LIMIAR).astype(np.uint8)
    pred_limpa = limpar_mascara(pred_bruta)

    return img, mask_gt_bin, pred_bruta, pred_limpa


def main():
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()

    with open(Path(SPLITS_DIR) / "test.txt", "r", encoding="utf-8") as f:
        stems_teste = [l.strip() for l in f if l.strip()]

    print(f"Aplicando limpeza pos-predicao em {len(stems_teste)} imagens do teste...\n")
    print(f"{'imagem':<40} {'dice_bruto':>11} {'dice_limpo':>11} {'diferenca':>11}")
    print("-" * 76)

    resultados = []
    for stem in stems_teste:
        img, mask_gt, pred_bruta, pred_limpa = carregar_e_prever(model, stem)
        dice_bruto = dice_score(pred_bruta, mask_gt)
        dice_limpo = dice_score(pred_limpa, mask_gt)
        resultados.append((stem, dice_bruto, dice_limpo, img, mask_gt, pred_bruta, pred_limpa))
        print(f"{stem[:38]:<40} {dice_bruto:>11.4f} {dice_limpo:>11.4f} {dice_limpo - dice_bruto:>+11.4f}")

    # --------- csv comparativo ---------
    csv_path = Path(OUTPUT_DIR) / "dice_antes_depois_limpeza.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "dice_bruto", "dice_limpo", "diferenca"])
        for stem, db, dl, *_ in resultados:
            writer.writerow([stem, f"{db:.4f}", f"{dl:.4f}", f"{dl - db:+.4f}"])
    print(f"\nCsv comparativo salvo em: {csv_path}")

    dices_brutos = [r[1] for r in resultados]
    dices_limpos = [r[2] for r in resultados]
    print(f"\nDice medio ANTES da limpeza:  {np.mean(dices_brutos):.4f}")
    print(f"Dice medio DEPOIS da limpeza: {np.mean(dices_limpos):.4f}")

    # --------- Figura comparativa: Original | Predicao bruta | Predicao limpa | Ground truth ---------
    n = len(resultados)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 4)

    for row, (stem, db, dl, img, mask_gt, pred_bruta, pred_limpa) in zip(axes, resultados):
        row[0].imshow(img, cmap="gray")
        row[0].set_title(f"{stem[:18]} - Original")
        row[0].axis("off")

        row[1].imshow(pred_bruta, cmap="gray")
        row[1].set_title(f"Bruta (Dice={db:.3f})")
        row[1].axis("off")

        row[2].imshow(pred_limpa, cmap="gray")
        row[2].set_title(f"Limpa (Dice={dl:.3f})")
        row[2].axis("off")

        row[3].imshow(mask_gt, cmap="gray")
        row[3].set_title("Ground truth")
        row[3].axis("off")

    plt.tight_layout()
    fig_path = Path(OUTPUT_DIR) / "figuras_comparativas_limpas.png"
    plt.savefig(fig_path, dpi=120)
    print(f"Figura comparativa (bruta vs limpa) salva em: {fig_path}")


if __name__ == "__main__":
    main()
