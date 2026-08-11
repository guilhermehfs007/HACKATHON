import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

# ------------------- CONFIG -------------------
IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/images"
ANNOTATIONS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/individual"
OUT_IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/images"
OUT_MASKS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/masks"

NOVO_TAMANHO = (1024, 512)   # (largura, altura)
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
# ------------------------------------------------

os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_MASKS_DIR, exist_ok=True)


def rasterizar_mascara(data: dict, img_info: dict) -> np.ndarray:
    """Desenha os poligonos do COCO json numa mascara binaria (0/255)."""
    width, height = img_info["width"], img_info["height"]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    anotacoes = [a for a in data.get("annotations", []) if a["image_id"] == img_info["id"]]
    for ann in anotacoes:
        for polygon in ann.get("segmentation", []):
            pontos = list(zip(polygon[0::2], polygon[1::2]))
            if len(pontos) >= 3:
                draw.polygon(pontos, fill=255)

    return np.array(mask)


def processar(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("images"):
        return None

    img_info = data["images"][0]
    file_name = img_info["file_name"]
    img_path = Path(IMAGES_DIR) / file_name
    if not img_path.exists():
        print(f"  [AVISO] imagem nao encontrada para {file_name}, pulando.")
        return None

    # --- Imagem: CLAHE + resize (AREA) ---
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    img_clahe = clahe.apply(img)
    img_final = cv2.resize(img_clahe, NOVO_TAMANHO, interpolation=cv2.INTER_AREA)

    # --- Mascara: rasterizar + resize (NEAREST) ---
    mask = rasterizar_mascara(data, img_info)
    mask_final = cv2.resize(mask, NOVO_TAMANHO, interpolation=cv2.INTER_NEAREST)

    # --- Salvar ---
    nome_base = Path(file_name).stem
    cv2.imwrite(str(Path(OUT_IMAGES_DIR) / f"{nome_base}.png"), img_final)
    cv2.imwrite(str(Path(OUT_MASKS_DIR) / f"{nome_base}_mask.png"), mask_final)
    return nome_base


if __name__ == "__main__":
    json_files = sorted(Path(ANNOTATIONS_DIR).glob("*.json"))
    print(f"Processando {len(json_files)} imagens...\n")

    processadas = 0
    for jf in json_files:
        resultado = processar(jf)
        if resultado:
            processadas += 1
            print(f"  OK: {resultado}")

    print(f"\nConcluido: {processadas}/{len(json_files)} imagens processadas com sucesso.")
    print(f"Imagens salvas em: {OUT_IMAGES_DIR}")
    print(f"Mascaras salvas em: {OUT_MASKS_DIR}")
