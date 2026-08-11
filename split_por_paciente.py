import json
import re
import csv
import random
from pathlib import Path
from collections import defaultdict

# ------------------- CONFIG -------------------
ANNOTATIONS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/individual"
PROCESSED_IMAGES_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/processed/images"
OUT_SPLITS_DIR = "/home/akinoriii/VSC/VSCPP/Hackathon/splits"

# Se voce tiver um csv confiavel de paciente, aponte aqui. Caso contrario, None.
MAPEAMENTO_MANUAL_CSV = None  # ex: "/home/.../Hackathon/patient_mapping.csv"

TREINO_FRAC = 0.70
VAL_FRAC = 0.15
TESTE_FRAC = 0.15
SEED = 42
# ------------------------------------------------


def extrair_paciente_id(file_name: str) -> str:
    """
    Ajustado ao padrao real observado no dataset, ex:
    "non_perio_1035-tif_0_png.rf.eaf763e0809f7cd8ff2d4689f273e4d88.png"
    -> o numero antes de "-tif" (aqui, "1035") e o identificador do exame
    original, antes do Roboflow reexportar/hashear o nome.
    """
    stem = Path(file_name).stem
    stem = re.sub(r"\.rf\..*$", "", stem)  # remove hash de exportacao Roboflow

    m = re.search(r"(\d+)-tif", stem)
    if m:
        return m.group(1)

    # fallback 1: qualquer sequencia de digitos isolada no nome
    m = re.search(r"(\d{3,})", stem)
    if m:
        return m.group(1)

    # fallback 2: tudo antes do primeiro "_"/"-" seguido de numero
    m = re.match(r"^([A-Za-z0-9]+)[_\-]", stem)
    if m:
        return m.group(1)

    return stem  # pior caso: cada imagem vira "paciente" unico


def carregar_mapeamento_manual(caminho_csv):
    mapa = {}
    with open(caminho_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapa[row["file_name"]] = row["patient_id"]
    return mapa


def main():
    random.seed(SEED)
    Path(OUT_SPLITS_DIR).mkdir(parents=True, exist_ok=True)

    json_files = sorted(Path(ANNOTATIONS_DIR).glob("*.json"))
    print(f"Total de arquivos de anotacao: {len(json_files)}")

    mapeamento_manual = carregar_mapeamento_manual(MAPEAMENTO_MANUAL_CSV) if MAPEAMENTO_MANUAL_CSV else {}

    registros = []  # (file_name, stem_processado, patient_id)
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("images"):
            continue
        file_name = data["images"][0]["file_name"]

        # so entra no split se a versao processada (preprocess_batch.py) existir
        stem_processado = Path(file_name).stem
        img_processada = Path(PROCESSED_IMAGES_DIR) / f"{stem_processado}.png"
        if not img_processada.exists():
            continue

        patient_id = mapeamento_manual.get(file_name) or extrair_paciente_id(file_name)
        registros.append((file_name, stem_processado, patient_id))

    print(f"Imagens com par processado encontrado: {len(registros)}")

    # --------- DEBUG: mostra a extracao pra voce validar visualmente ---------
    print("\n=== DEBUG: primeiros 15 patient_id extraidos ===")
    print(f"{'file_name':<40} {'patient_id_extraido':<25}")
    for file_name, _, pid in registros[:15]:
        print(f"{file_name[:38]:<40} {pid:<25}")
    print("=================================================\n")

    # --------- Agrupa por paciente ---------
    por_paciente = defaultdict(list)
    for file_name, stem_processado, pid in registros:
        por_paciente[pid].append((file_name, stem_processado))

    pacientes = list(por_paciente.keys())
    random.shuffle(pacientes)
    n_pacientes = len(pacientes)
    print(f"Numero de pacientes distintos: {n_pacientes}")
    if n_pacientes == len(registros):
        print(">> ATENCAO: cada imagem virou um paciente diferente. A heuristica")
        print("   provavelmente NAO esta identificando pacientes de verdade -- revise")
        print("   extrair_paciente_id() antes de confiar neste split.\n")

    # --------- Split por PACIENTE (nao por imagem) ---------
    n_treino = max(1, round(n_pacientes * TREINO_FRAC))
    n_val = max(1, round(n_pacientes * VAL_FRAC))
    pacientes_treino = set(pacientes[:n_treino])
    pacientes_val = set(pacientes[n_treino:n_treino + n_val])
    pacientes_teste = set(pacientes[n_treino + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for pid, imgs in por_paciente.items():
        if pid in pacientes_treino:
            destino = "train"
        elif pid in pacientes_val:
            destino = "val"
        else:
            destino = "test"
        for file_name, stem_processado in imgs:
            splits[destino].append(stem_processado)

    # --------- Verificacao de vazamento (nenhum paciente em 2 splits) ---------
    assert not (pacientes_treino & pacientes_val)
    assert not (pacientes_treino & pacientes_teste)
    assert not (pacientes_val & pacientes_teste)
    print("OK: nenhum paciente aparece em mais de um split.\n")

    # --------- Salva manifests (listas de stems, uma por linha) ---------
    for nome, stems in splits.items():
        out_path = Path(OUT_SPLITS_DIR) / f"{nome}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(stems)))
        print(f"{nome:<6}: {len(stems):>4} imagens / {out_path}")

    # --------- Salva csv de auditoria (pra mostrar no pitch se pedirem) ---------
    auditoria_path = Path(OUT_SPLITS_DIR) / "patient_split_auditoria.csv"
    with open(auditoria_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "stem_processado", "patient_id", "split"])
        for pid, imgs in por_paciente.items():
            destino = "train" if pid in pacientes_treino else ("val" if pid in pacientes_val else "test")
            for file_name, stem_processado in imgs:
                writer.writerow([file_name, stem_processado, pid, destino])
    print(f"\nAuditoria completa salva em: {auditoria_path}")
    print("(mostra, por imagem, qual paciente e qual split -- guarda isso pro checklist)")


if __name__ == "__main__":
    main()
