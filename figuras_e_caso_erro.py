"""
Figuras comparativas + selecao do caso de erro -- conjunto de TESTE
=======================================================================
Roda o modelo no conjunto de teste (o mesmo usado pra metrica final),
calcula Dice por imagem, e gera:
  1. Uma grade com N exemplos (original | predicao | ground truth)
  2. Uma figura separada so com o PIOR caso (menor Dice) -- e o "caso de
     erro apresentado por voces mesmos" que o checklist pede
  3. Um csv com o Dice de cada imagem do teste, ordenado (serve tambem
     pra escolher outros exemplos "bons" pra grade, sem cherry-pick
     escondido -- mostra o csv inteiro se alguem perguntar)
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
N_EXEMPLOS_GRADE = 4   # quantos exemplos "normais" mostrar (>=3 pedido no checklist)
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


def dice_score(pred_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    tp = np.logical_and(pred_bin, mask_bin).sum()
    return (2 * tp + 1e-7) / (pred_bin.sum() + mask_bin.sum() + 1e-7)


def carregar_e_prever(model, stem):
    img = cv2.imread(str(Path(PROCESSED_IMAGES_DIR) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    mask_gt = cv2.imread(str(Path(PROCESSED_MASKS_DIR) / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    mask_gt_bin = (mask_gt > 0).astype(np.uint8)

    tensor = transform(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    pred_bin = (pred > LIMIAR).astype(np.uint8)

    return img, mask_gt_bin, pred_bin, dice_score(pred_bin, mask_gt_bin)


def montar_linha(axes_row, img, mask_gt, pred, titulo_prefixo, dice):
    axes_row[0].imshow(img, cmap="gray")
    axes_row[0].set_title(f"{titulo_prefixo} - Original")
    axes_row[0].axis("off")

    axes_row[1].imshow(pred, cmap="gray")
    axes_row[1].set_title(f"Predicao (Dice={dice:.3f})")
    axes_row[1].axis("off")

    axes_row[2].imshow(mask_gt, cmap="gray")
    axes_row[2].set_title("Ground truth")
    axes_row[2].axis("off")


def main():
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()

    with open(Path(SPLITS_DIR) / "test.txt", "r", encoding="utf-8") as f:
        stems_teste = [l.strip() for l in f if l.strip()]

    print(f"Avaliando {len(stems_teste)} imagens do conjunto de teste...")

    resultados = []  # (stem, dice, img, mask_gt, pred)
    for stem in stems_teste:
        img, mask_gt, pred, dice = carregar_e_prever(model, stem)
        resultados.append((stem, dice, img, mask_gt, pred))

    resultados.sort(key=lambda r: r[1])  # ordena por Dice crescente (pior -> melhor)

    # --------- csv com Dice por imagem (auditoria, sem cherry-pick escondido) ---------
    csv_path = Path(OUTPUT_DIR) / "dice_por_imagem_teste.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "dice"])
        for stem, dice, *_ in resultados:
            writer.writerow([stem, f"{dice:.4f}"])
    print(f"Dice por imagem salvo em: {csv_path}")

    # --------- Grade de N exemplos "representativos" (espacados pela distribuicao de Dice) ---------
    indices = np.linspace(0, len(resultados) - 1, N_EXEMPLOS_GRADE).astype(int)
    exemplos = [resultados[i] for i in indices]

    fig, axes = plt.subplots(N_EXEMPLOS_GRADE, 3, figsize=(10, 3.2 * N_EXEMPLOS_GRADE))
    if N_EXEMPLOS_GRADE == 1:
        axes = axes.reshape(1, 3)
    for row, (stem, dice, img, mask_gt, pred) in zip(axes, exemplos):
        montar_linha(row, img, mask_gt, pred, stem[:18], dice)
    plt.tight_layout()
    grade_path = Path(OUTPUT_DIR) / "figuras_comparativas.png"
    plt.savefig(grade_path, dpi=120)
    print(f"Grade comparativa salva em: {grade_path}")

    # --------- Caso de erro: o PIOR Dice do conjunto de teste ---------
    stem_pior, dice_pior, img_pior, mask_pior, pred_pior = resultados[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    montar_linha(axes, img_pior, mask_pior, pred_pior, stem_pior[:25], dice_pior)
    plt.suptitle(f"Caso de erro selecionado -- pior Dice do teste ({dice_pior:.3f})")
    plt.tight_layout()
    erro_path = Path(OUTPUT_DIR) / "caso_de_erro.png"
    plt.savefig(erro_path, dpi=120)
    print(f"Caso de erro salvo em: {erro_path}  (stem: {stem_pior}, Dice: {dice_pior:.4f})")

    dices = [r[1] for r in resultados]
    print(f"\nDice no teste -- media: {np.mean(dices):.4f}  min: {np.min(dices):.4f}  max: {np.max(dices):.4f}")


if __name__ == "__main__":
    main()
