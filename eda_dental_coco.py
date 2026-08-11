import os
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ------------------- CONFIG: AJUSTE AQUI -------------------
IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/images"
ANNOTATIONS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/individual"
OUTPUT_DIR = "eda_output"
# -------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

json_files = sorted(Path(ANNOTATIONS_DIR).glob("*.json"))
print(f"Total de arquivos de anotacao encontrados: {len(json_files)}")

# --------- DEBUG: comparar nomes esperados (JSON) vs arquivos reais ---------
DEBUG_NOMES = True
if DEBUG_NOMES and json_files:
    with open(json_files[0], "r", encoding="utf-8") as f:
        exemplo = json.load(f)
    nome_esperado = exemplo["images"][0]["file_name"]
    arquivos_reais = sorted(os.listdir(IMAGES_DIR))[:10]
    print("\n=== DEBUG DE NOMES ===")
    print(f"Nome esperado (vindo do JSON): {nome_esperado}")
    print(f"Primeiros 10 arquivos reais em IMAGES_DIR: {arquivos_reais}")
    print("=======================\n")
# ------------------------------------------------------------------------

proporcoes_dente = []
n_instancias_por_imagem = []
medias_brilho = []
sizes = []
sem_imagem_correspondente = []

for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Estrutura padrao Roboflow: data["images"][0]["file_name"], data["annotations"]
    if not data.get("images"):
        continue
    img_info = data["images"][0]
    file_name = img_info["file_name"]
    width, height = img_info["width"], img_info["height"]

    img_path = Path(IMAGES_DIR) / file_name
    if not img_path.exists():
        sem_imagem_correspondente.append(file_name)
        continue

    sizes.append((width, height))

    # --- Rasterizar todos os poligonos desta imagem numa mascara ---
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    anotacoes_desta_imagem = [a for a in data.get("annotations", [])
                               if a["image_id"] == img_info["id"]]
    n_instancias_por_imagem.append(len(anotacoes_desta_imagem))

    for ann in anotacoes_desta_imagem:
        seg = ann.get("segmentation", [])
        for polygon in seg:
            # polygon vem como [x1, y1, x2, y2, ...] -> converter em pares
            pontos = list(zip(polygon[0::2], polygon[1::2]))
            if len(pontos) >= 3:
                draw.polygon(pontos, fill=255)

    mask_arr = np.array(mask)
    proporcoes_dente.append((mask_arr > 0).mean())

    # --- Brilho da imagem original ---
    with Image.open(img_path) as im:
        arr = np.array(im.convert("L")).astype(np.float32)
        medias_brilho.append(arr.mean())

# --------- Relatorio ---------
print(f"\nAnotacoes SEM imagem correspondente: {len(sem_imagem_correspondente)}")
if sem_imagem_correspondente:
    print("  Exemplos:", sem_imagem_correspondente[:5])

pares_validos = len(proporcoes_dente)
print(f"Pares validos processados: {pares_validos}")

proporcoes_dente = np.array(proporcoes_dente)
n_instancias_por_imagem = np.array(n_instancias_por_imagem)
medias_brilho = np.array(medias_brilho)
sizes_arr = np.array(sizes)

print("\n--- Resolucao das imagens ---")
print(f"Largura - min: {sizes_arr[:,0].min()} max: {sizes_arr[:,0].max()} media: {sizes_arr[:,0].mean():.0f}")
print(f"Altura  - min: {sizes_arr[:,1].min()} max: {sizes_arr[:,1].max()} media: {sizes_arr[:,1].mean():.0f}")

print("\n--- Balanceamento de classes (% de pixels = DENTE) ---")
print(f"Media: {proporcoes_dente.mean()*100:.2f}%  min: {proporcoes_dente.min()*100:.2f}%  max: {proporcoes_dente.max()*100:.2f}%")

print("\n--- Numero de dentes (instancias) por imagem ---")
print(f"Media: {n_instancias_por_imagem.mean():.1f}  min: {n_instancias_por_imagem.min()}  max: {n_instancias_por_imagem.max()}")

print("\n--- Variacao de exposicao (brilho medio por imagem) ---")
print(f"min: {medias_brilho.min():.1f}  max: {medias_brilho.max():.1f}  desvio entre imagens: {medias_brilho.std():.1f}")

# --------- Graficos ---------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(proporcoes_dente * 100, bins=20, color="steelblue")
axes[0].set_title("% de pixels = DENTE")

axes[1].hist(n_instancias_por_imagem, bins=range(0, int(n_instancias_por_imagem.max())+2), color="indianred")
axes[1].set_title("Nº de dentes por imagem")

axes[2].hist(medias_brilho, bins=20, color="darkorange")
axes[2].set_title("Brilho médio por imagem")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "eda_summary_coco.png")
plt.savefig(out_path, dpi=120)
print(f"\nGraficos salvos em: {out_path}")