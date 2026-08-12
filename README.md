# HACKATHON Alliage 2026 — Segmentação de Imagem com IA

Segmentação semântica (dente vs. fundo) em radiografias panorâmicas
odontológicas, usando um Unet com encoder ResNet34 pré-treinado
(transfer learning).

## O problema

Hoje, delimitar estruturas numa radiografia panorâmica é feito manualmente,
clique por clique, e o resultado varia entre profissionais. Isso impede
comparação estável entre exames do mesmo paciente. O objetivo deste projeto
é um modelo que receba a radiografia e devolva a máscara de segmentação
automaticamente, com qualidade medida.

## Estrutura do repositório

```
split_por_paciente.py       # gera splits/train.txt, val.txt, test.txt (split por paciente)
preprocess_batch.py         # CLAHE + resize em lote, gera processed/images e processed/masks
preprocess_demo.py          # demo visual do pre-processamento numa imagem so
augmentation_demo.py        # demo visual do pipeline de aumento de dados
eda_dental_coco.py          # analise exploratoria das anotacoes COCO
verificar_alinhamento.py    # overlay visual imagem+mascara pra conferir alinhamento
verificar_precisao_mascara.py  # mede erro de area introduzido pelo resize das mascaras
treinar_modelo.py           # treino local (CPU/GPU) do Unet + ResNet34
treinar_modelo_colab.ipynb  # mesma logica de treino, adaptada pra rodar no Google Colab (GPU gratis)
figuras_e_caso_erro.py      # gera figuras comparativas + seleciona o pior caso do teste
limpar_predicoes.py         # limpeza morfologica pos-predicao (sem retreinar)
inferencia.py                # roda o modelo em imagens sem anotacao + mede tempo por imagem
.gitignore
```

## Requisitos

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations opencv-python
pip install matplotlib numpy
```

Se for treinar com GPU local (NVIDIA), instale o PyTorch seguindo as
instruções específicas em pytorch.org/get-started/locally — o comando acima
instala a versão CPU-only.

## Como rodar (ordem de execução)

1. **`python preprocess_batch.py`** — gera `processed/images` e
   `processed/masks` a partir de `images/` e `individual/` (anotações COCO).
2. **`python split_por_paciente.py`** — separa treino/validação/teste
   agrupando por paciente (evita vazamento de dados entre conjuntos). Gera
   `splits/train.txt`, `val.txt`, `test.txt` e um csv de auditoria.
3. **`python treinar_modelo.py`** (ou o notebook `treinar_modelo_colab.ipynb`
   no Google Colab, se precisar de GPU) — treina o modelo, salva o melhor
   checkpoint em `treino_output/melhor_modelo.pt` por Dice de validação.
4. **`python figuras_e_caso_erro.py`** — avalia no conjunto de teste, gera
   as figuras comparativas e seleciona o pior caso (caso de erro).
5. **`python limpar_predicoes.py`** — aplica limpeza morfológica pós-predição
   (opcional, não retreina nada) e compara Dice antes/depois.
6. **`python inferencia.py`** — roda o modelo em imagens sem anotação e mede
   tempo de inferência por imagem.

## Métrica e resultado

**Dice**, medido no conjunto de **teste** (nunca visto no treino):

| | Dice médio |
|---|---|
| Predição bruta | ~0.45 |
| Após limpeza morfológica pós-predição | ~0.72 |

O ganho da limpeza vem principalmente da remoção de ruído espalhado pela
imagem (falsos positivos fora da região dentária) — a forma predita ainda é
uma região única aproximada, não o contorno de cada dente separado. Ou seja,
o projeto entrega **segmentação semântica** (dente vs. fundo), que é o
obrigatório do desafio; segmentação por instância (dente por dente) é o
diferencial listado no enunciado e não foi implementado.

## Limitações conhecidas

- **Dataset pequeno**: 30 imagens anotadas no total (21 treino / 4 validação
  / 5 teste). O Dice medido em 5 imagens tem alta variância.
- **Poucas épocas de treino**: por restrição de tempo do hackathon, o modelo
  foi treinado por poucas épocas. Isso é visível nos casos de erro — o
  modelo aprendeu a localizar a região dentária, mas ainda não aprendeu
  contornos finos por dente.
- **Split por paciente**: confirmado 1 imagem por paciente neste dataset,
  então o split por paciente é equivalente a um split direto por imagem —
  documentado em `splits/patient_split_auditoria.csv`.
