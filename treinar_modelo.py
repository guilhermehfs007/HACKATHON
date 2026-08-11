"""
Treino: segmentacao semantica (dente vs fundo) -- Unet + ResNet34
====================================================================
Requer: pip install torch torchvision segmentation-models-pytorch albumentations opencv-python

Le os manifests gerados por split_por_paciente.py (splits/train.txt, val.txt,
test.txt), treina um Unet com encoder ResNet34 pre-treinado (transfer
learning), e no final reporta a metrica no conjunto de TESTE (nao no de
validacao -- validacao serve so pra escolher o melhor checkpoint).

Escolha de funcao de perda: Dice Loss + BCE combinadas. So BCE tende a
ignorar a classe minoritaria (fundo domina os pixels); Dice Loss otimiza
sobreposicao diretamente, que e a metrica que o desafio pede no final.
Combinar as duas costuma estabilizar o treino melhor que usar so uma.
"""

import os
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
PROCESSED_IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/images"
PROCESSED_MASKS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/masks"
SPLITS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/splits"
OUTPUT_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/treino_output"

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
EPOCHS = 60
BATCH_SIZE = 4          # imagens 1024x512 pesam -- suba se sua GPU aguentar
LR = 1e-4
PATIENCE_EARLY_STOP = 10  # epocas sem melhora no Dice de val antes de parar
IMG_SIZE = (1024, 512)    # deve bater com o que preprocess_batch.py gerou
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")


def ler_manifest(nome):
    path = Path(SPLITS_DIR) / f"{nome}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


class DentalDataset(Dataset):
    def __init__(self, stems, transform=None):
        self.stems = stems
        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = cv2.imread(str(Path(PROCESSED_IMAGES_DIR) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(Path(PROCESSED_MASKS_DIR) / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        return img, mask, stem


# mesma logica de augmentation do augmentation_demo.py, so que empacotada
# com normalizacao + conversao pra tensor (necessario pro treino)
transform_treino = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-12, 12), translate_percent=(0.0, 0.05), scale=(0.95, 1.05), p=0.7),
    A.ElasticTransform(alpha=40, sigma=6, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

# val/teste: SEM augmentation -- so normalizacao, pra medir desempenho real
transform_eval = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


def calcular_dice_iou(pred_logits, mask, threshold=0.5):
    """Dice e IoU binarios pra um batch. Retorna medias do batch."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    tp = (pred * mask).sum(dim=(1, 2, 3))
    fp = (pred * (1 - mask)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * mask).sum(dim=(1, 2, 3))
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    return dice.mean().item(), iou.mean().item()


def rodar_epoca(model, loader, criterion_dice, criterion_bce, optimizer=None):
    treinando = optimizer is not None
    model.train() if treinando else model.eval()

    perda_total, dice_total, iou_total, n_batches = 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(treinando):
        for img, mask, _ in loader:
            img, mask = img.to(device), mask.to(device)

            if treinando:
                optimizer.zero_grad()

            pred = model(img)
            perda = criterion_dice(pred, mask) + criterion_bce(pred, mask)

            if treinando:
                perda.backward()
                optimizer.step()

            dice, iou = calcular_dice_iou(pred.detach(), mask)
            perda_total += perda.item()
            dice_total += dice
            iou_total += iou
            n_batches += 1

    return perda_total / n_batches, dice_total / n_batches, iou_total / n_batches


def main():
    stems_treino = ler_manifest("train")
    stems_val = ler_manifest("val")
    stems_teste = ler_manifest("test")
    print(f"Treino: {len(stems_treino)} | Val: {len(stems_val)} | Teste: {len(stems_teste)}")

    ds_treino = DentalDataset(stems_treino, transform=transform_treino)
    ds_val = DentalDataset(stems_val, transform=transform_eval)
    ds_teste = DentalDataset(stems_teste, transform=transform_eval)

    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    dl_teste = DataLoader(ds_teste, batch_size=1, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=1,
    ).to(device)

    criterion_dice = smp.losses.DiceLoss(mode="binary")
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    historico = {"treino_perda": [], "val_perda": [], "val_dice": [], "val_iou": []}
    melhor_dice_val = -1.0
    epocas_sem_melhora = 0
    checkpoint_path = Path(OUTPUT_DIR) / "melhor_modelo.pt"

    for epoca in range(1, EPOCHS + 1):
        t0 = time.time()
        perda_treino, dice_treino, iou_treino = rodar_epoca(model, dl_treino, criterion_dice, criterion_bce, optimizer)
        perda_val, dice_val, iou_val = rodar_epoca(model, dl_val, criterion_dice, criterion_bce, optimizer=None)
        scheduler.step(dice_val)

        historico["treino_perda"].append(perda_treino)
        historico["val_perda"].append(perda_val)
        historico["val_dice"].append(dice_val)
        historico["val_iou"].append(iou_val)

        dt = time.time() - t0
        print(f"[{epoca:03d}/{EPOCHS}] treino_perda={perda_treino:.4f} | "
              f"val_perda={perda_val:.4f} val_dice={dice_val:.4f} val_iou={iou_val:.4f} | {dt:.1f}s")

        if dice_val > melhor_dice_val:
            melhor_dice_val = dice_val
            epocas_sem_melhora = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> novo melhor modelo salvo (val_dice={dice_val:.4f})")
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= PATIENCE_EARLY_STOP:
                print(f"\nEarly stopping: sem melhora ha {PATIENCE_EARLY_STOP} epocas.")
                break

    # --------- Curvas de treino ---------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(historico["treino_perda"], label="treino")
    axes[0].plot(historico["val_perda"], label="val")
    axes[0].set_title("Perda (Dice+BCE)")
    axes[0].legend()
    axes[1].plot(historico["val_dice"], label="Dice (val)")
    axes[1].plot(historico["val_iou"], label="IoU (val)")
    axes[1].set_title("Metricas de validacao")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "curvas_treino.png", dpi=120)

    # --------- Avaliacao final no conjunto de TESTE (metrica que vai pro relatorio) ---------
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    tempos_inferencia = []
    dices, ious = [], []
    with torch.no_grad():
        for img, mask, stem in dl_teste:
            img, mask = img.to(device), mask.to(device)
            t0 = time.time()
            pred = model(img)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tempos_inferencia.append(time.time() - t0)
            dice, iou = calcular_dice_iou(pred, mask)
            dices.append(dice)
            ious.append(iou)

    dice_final = float(np.mean(dices))
    iou_final = float(np.mean(ious))
    tempo_medio_ms = float(np.mean(tempos_inferencia)) * 1000

    print("\n" + "=" * 60)
    print("RESULTADO FINAL -- conjunto de TESTE (nunca visto no treino)")
    print(f"  Dice: {dice_final:.4f}")
    print(f"  IoU:  {iou_final:.4f}")
    print(f"  Tempo medio de inferencia por imagem: {tempo_medio_ms:.1f} ms  (device: {device})")
    print("=" * 60)

    with open(Path(OUTPUT_DIR) / "resultado_final_teste.txt", "w", encoding="utf-8") as f:
        f.write("Metrica reportada no conjunto de TESTE (held-out, nunca visto no treino)\n")
        f.write(f"Dice: {dice_final:.4f}\n")
        f.write(f"IoU: {iou_final:.4f}\n")
        f.write(f"Tempo medio de inferencia por imagem: {tempo_medio_ms:.1f} ms (device: {device})\n")
        f.write(f"N de imagens no teste: {len(stems_teste)}\n")

    print(f"\nModelo salvo em: {checkpoint_path}")
    print(f"Relatorio salvo em: {Path(OUTPUT_DIR) / 'resultado_final_teste.txt'}")


if __name__ == "__main__":
    main()
