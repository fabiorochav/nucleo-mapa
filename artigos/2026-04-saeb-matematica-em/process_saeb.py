"""
process_saeb.py
---------------
Processa os microdados do Saeb 2023 (3ª/4ª série do Ensino Médio)
e gera as tabelas resumidas de distribuição por nível de proficiência
para Matemática (dist_mt.csv) e Língua Portuguesa (dist_lp.csv).

Dependências: pandas, numpy
    pip install pandas numpy

Uso:
    Coloque microdados_saeb_2023.zip na mesma pasta deste script e execute:
    python process_saeb.py

Metodologia de classificação (Francisco Soares / IDESP):
    Abaixo do Básico : < 275 pontos
    Básico           : 275 – 350
    Adequado         : 350 – 400
    Avançado         : > 400

Fonte: Microdados Saeb 2023 – Inep
"""

import zipfile, io, os
import pandas as pd
import numpy as np

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH  = os.path.join(BASE_DIR, "microdados_saeb_2023.zip")
CSV_INNER = "MICRODADOS_SAEB_2023/DADOS/TS_ALUNO_34EM.csv"

OUT_MT = os.path.join(BASE_DIR, "dist_mt.csv")
OUT_LP = os.path.join(BASE_DIR, "dist_lp.csv")

# ── Mapeamento UF ─────────────────────────────────────────────────────────────
UF_MAP = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}

NIVEIS    = ["Abaixo do Básico", "Básico", "Adequado", "Avançado"]
CUTOFFS   = [(-float("inf"), 275), (275, 350), (350, 400), (400, float("inf"))]

COLS = [
    "ID_UF", "IN_PUBLICA",
    "IN_PROFICIENCIA_MT", "PROFICIENCIA_MT_SAEB", "PESO_ALUNO_MT",
    "IN_PROFICIENCIA_LP", "PROFICIENCIA_LP_SAEB", "PESO_ALUNO_LP",
]

# ── Funções auxiliares ────────────────────────────────────────────────────────
def classify(score):
    for nivel, (lo, hi) in zip(NIVEIS, CUTOFFS):
        if lo <= score < hi:
            return nivel
    return None

def dist_ponderada(df, profic_col, peso_col):
    """Distribuição percentual ponderada por nível de proficiência, por UF + BR."""
    rows = []
    for uf_code, grp in df.groupby("ID_UF"):
        total = grp[peso_col].sum()
        d = grp.groupby("nivel")[peso_col].sum() / total * 100
        rows.append({"uf": UF_MAP[uf_code],
                     **{n: round(d.get(n, 0), 1) for n in NIVEIS}})
    # Nacional (BR)
    total_br = df[peso_col].sum()
    d_br = df.groupby("nivel")[peso_col].sum() / total_br * 100
    rows.append({"uf": "BR", **{n: round(d_br.get(n, 0), 1) for n in NIVEIS}})
    return pd.DataFrame(rows)

# ── Leitura ───────────────────────────────────────────────────────────────────
print("Lendo microdados (pode levar alguns segundos)...")
with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(CSV_INNER) as f:
        raw = pd.read_csv(
            io.TextIOWrapper(f, encoding="latin-1"),
            sep=";",
            usecols=COLS,
            low_memory=True,
        )

print(f"  Total de registros : {len(raw):>10,}")
pub = raw[raw["IN_PUBLICA"] == 1].copy()
print(f"  Rede pública (EM)  : {len(pub):>10,}")

# ── Matemática ────────────────────────────────────────────────────────────────
mt = pub[pub["IN_PROFICIENCIA_MT"] == 1].dropna(
    subset=["PROFICIENCIA_MT_SAEB", "PESO_ALUNO_MT"]
).copy()
mt["nivel"] = mt["PROFICIENCIA_MT_SAEB"].apply(classify)

df_mt = dist_ponderada(mt, "PROFICIENCIA_MT_SAEB", "PESO_ALUNO_MT")
df_mt.to_csv(OUT_MT, index=False)
print(f"\nMatemática  → {OUT_MT}")
print(df_mt.to_string(index=False))

# ── Língua Portuguesa ─────────────────────────────────────────────────────────
lp = pub[pub["IN_PROFICIENCIA_LP"] == 1].dropna(
    subset=["PROFICIENCIA_LP_SAEB", "PESO_ALUNO_LP"]
).copy()
lp["nivel"] = lp["PROFICIENCIA_LP_SAEB"].apply(classify)

df_lp = dist_ponderada(lp, "PROFICIENCIA_LP_SAEB", "PESO_ALUNO_LP")
df_lp.to_csv(OUT_LP, index=False)
print(f"\nLíngua Portuguesa → {OUT_LP}")
print(df_lp.to_string(index=False))

print("\nConcluído.")
