import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

# ------------------- CONFIG -------------------
IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/images"
ANNOTATIONS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/individual"
NOVO_TAMANHO = (1024, 512)
# ------------------------------------------------


def rasterizar_mascara(data: dict, img_info: dict) -> np.ndarray:
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


json_files = sorted(Path(ANNOTATIONS_DIR).glob("*.json"))
erros = []

print(f"{'Imagem':<45} {'% original':>12} {'% final':>12} {'erro (p.p.)':>12}")
print("-" * 85)

for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("images"):
        continue

    img_info = data["images"][0]
    file_name = img_info["file_name"]
    img_path = Path(IMAGES_DIR) / file_name
    if not img_path.exists():
        continue

    mask_original = rasterizar_mascara(data, img_info)
    pct_original = (mask_original > 0).mean() * 100

    mask_final = cv2.resize(mask_original, NOVO_TAMANHO, interpolation=cv2.INTER_NEAREST)
    pct_final = (mask_final > 0).mean() * 100

    erro = abs(pct_final - pct_original)
    erros.append(erro)

    print(f"{Path(file_name).stem[:43]:<45} {pct_original:>11.2f}% {pct_final:>11.2f}% {erro:>11.3f}")

erros = np.array(erros)
print("-" * 85)
print(f"\nErro medio de area (resize): {erros.mean():.3f} pontos percentuais")
print(f"Erro maximo encontrado:      {erros.max():.3f} pontos percentuais")
print(f"Erro minimo encontrado:      {erros.min():.3f} pontos percentuais")

if erros.mean() < 1.0:
    print("\n>> Erro medio abaixo de 1 p.p. -- o resize esta preservando bem a area da mascara.")
else:
    print("\n>> ATENCAO: erro medio alto -- vale revisar o metodo de resize das mascaras.")
