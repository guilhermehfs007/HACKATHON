"""
Inferencia -- roda o modelo treinado nas imagens SEM anotacao
=================================================================
Usa o mesmo pre-processamento (CLAHE + resize) do preprocess_batch.py,
porque o modelo foi treinado em cima disso -- se voce mandar a imagem
crua pro modelo, o resultado degrada.

Salva: mascara predita (png) + overlay visual + tempo de inferencia por
imagem num csv (pro item "tempo de inferencia medido" do checklist).
"""

import os
import csv
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ------------------- CONFIG -------------------
IMAGENS_SEM_ANOTACAO_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/images_sem_anotacao"  # AJUSTE pro caminho real
CHECKPOINT_PATH = "/home/akinoriii/VSC/VSCPP/Hackathon/treino_output/melhor_modelo.pt"
OUTPUT_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/inferencia_output"

ENCODER = "resnet34"
IMG_SIZE = (1024, 512)   # (largura, altura) -- igual ao preprocess_batch.py
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
LIMIAR = 0.5             # threshold de binarizacao da predicao
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(Path(OUTPUT_DIR) / "mascaras", exist_ok=True)
os.makedirs(Path(OUTPUT_DIR) / "overlays", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


def preprocessar(img_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    img_clahe = clahe.apply(img_gray)
    return cv2.resize(img_clahe, IMG_SIZE, interpolation=cv2.INTER_AREA)


def main():
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    imagens = sorted(f for f in Path(IMAGENS_SEM_ANOTACAO_DIR).iterdir() if f.suffix.lower() in exts)
    print(f"Imagens sem anotacao encontradas: {len(imagens)}")

    resultados_tempo = []

    with torch.no_grad():
        for img_path in imagens:
            img_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_original is None:
                print(f"  [AVISO] nao consegui abrir {img_path.name}, pulando.")
                continue

            img_proc = preprocessar(img_original)
            tensor = transform(image=img_proc)["image"].unsqueeze(0).to(device)

            t0 = time.time()
            pred = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tempo_ms = (time.time() - t0) * 1000

            mask_pred = (torch.sigmoid(pred)[0, 0].cpu().numpy() > LIMIAR).astype(np.uint8) * 255

            nome_base = img_path.stem
            cv2.imwrite(str(Path(OUTPUT_DIR) / "mascaras" / f"{nome_base}_pred.png"), mask_pred)

            # overlay: imagem processada + mascara predita em vermelho
            img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
            overlay = img_rgb.copy()
            overlay[mask_pred > 0] = [255, 0, 0]
            resultado = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
            cv2.imwrite(str(Path(OUTPUT_DIR) / "overlays" / f"{nome_base}_overlay.png"), resultado)

            resultados_tempo.append((nome_base, tempo_ms))
            print(f"  {nome_base}: {tempo_ms:.1f} ms")

    # --------- Resumo de tempo de inferencia ---------
    tempos = [t for _, t in resultados_tempo]
    csv_path = Path(OUTPUT_DIR) / "tempos_inferencia.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imagem", "tempo_ms"])
        writer.writerows(resultados_tempo)

    print("\n" + "=" * 50)
    print(f"Tempo medio de inferencia: {np.mean(tempos):.1f} ms  (device: {device})")
    print(f"Tempo min/max: {np.min(tempos):.1f} / {np.max(tempos):.1f} ms")
    print(f"Detalhado em: {csv_path}")
    print(f"Mascaras em: {Path(OUTPUT_DIR) / 'mascaras'}")
    print(f"Overlays em: {Path(OUTPUT_DIR) / 'overlays'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
