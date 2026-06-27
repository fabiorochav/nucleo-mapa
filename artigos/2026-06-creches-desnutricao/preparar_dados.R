# ============================================================
# preparar_dados.R
# Roda UMA VEZ antes de renderizar o artigo.
# Gera os arquivos que o index.qmd vai carregar:
#   - sp_muni_tratamento.geojson   (mapa interativo)
#   - cs_desnut.rds                (event study C&S)
#   - desc_artigo.rds              (descritivas para a tabela)
#
# Execute com:
#   setwd("artigos/2026-06-creches-desnutricao")
#   source("preparar_dados.R")
# ============================================================

library(dplyr)
library(tidyr)
library(geobr)
library(sf)
library(did)

# ── 1. Painel DiD (gerado pelo script principal de análise) ───
# Assumindo que você rodou did_creches_sisvan_sp.R antes e o
# painel está disponível em analise_desnutricao/.
# Ajuste o caminho conforme necessário.
PAINEL_PATH <- "C:/Users/rocha/Documents/Acessados/Pessoal/exercicios-papers/analise_desnutricao/painel_did_sisvan_sp.rds"

if (!file.exists(PAINEL_PATH)) {
  stop("Painel não encontrado. Rode did_creches_sisvan_sp.R primeiro.\nCaminho esperado: ", PAINEL_PATH)
}

painel_did <- readRDS(PAINEL_PATH)

# ── 2. Geodata — municípios de SP ─────────────────────────────
message("Baixando shapefile de municípios de SP via geobr...")
geo_sp <- read_municipality(code_muni = "SP", year = 2024, showProgress = FALSE)

# Código de 6 dígitos para join
geo_sp <- geo_sp |>
  mutate(code_muni_6d = as.integer(substr(as.character(code_muni), 1, 6)))

# ── 3. Join com tratamento ────────────────────────────────────
tratamento <- painel_did |>
  filter(n_acompanhamentos >= 30) |>
  distinct(code_muni_6d, ano_tratamento) |>
  mutate(
    grupo = case_when(
      ano_tratamento == 0 ~ "Nunca tratado",
      TRUE ~ as.character(as.integer(ano_tratamento))
    )
  )

niveis_coorte <- c("2015","2016","2017","2018","2019","2020","2021","2022","2023","Nunca tratado")

geo_mapa <- geo_sp |>
  left_join(tratamento, by = "code_muni_6d") |>
  mutate(
    ano_tratamento = replace_na(as.integer(ano_tratamento), 0L),
    Coorte = factor(
      if_else(ano_tratamento == 0, "Nunca tratado", as.character(ano_tratamento)),
      levels = niveis_coorte
    )
  ) |>
  sf::st_transform(4326)

saveRDS(geo_mapa, "geo_mapa.rds")
message("✓ geo_mapa.rds salvo")

# ── 4. Estimar C&S e salvar objetos ──────────────────────────
MIN_E <- -4L
MAX_E <-  5L

painel_cs <- painel_did |>
  filter(
    n_acompanhamentos >= 30,
    !is.na(pct_desnutricao),
    ano_tratamento != 2016
  ) |>
  group_by(id_muni) |>
  filter(ano_tratamento == 0 | min(ano) < ano_tratamento) |>
  ungroup()

message("Estimando C&S (att_gt)... pode demorar ~1 min")
cs_desnut <- att_gt(
  yname         = "pct_desnutricao",
  tname         = "ano",
  idname        = "id_muni",
  gname         = "ano_tratamento",
  xformla       = NULL,
  data          = painel_cs,
  control_group = "notyettreated",
  anticipation  = 0,
  base_period   = "universal",
  est_method    = "reg",
  bstrap        = TRUE,
  cband         = TRUE
)

es_desnut <- aggte(cs_desnut, type = "dynamic",
                   min_e = MIN_E, max_e = MAX_E,
                   bstrap = TRUE, cband = TRUE)

att_simples <- aggte(cs_desnut, type = "simple",
                     bstrap = TRUE, cband = TRUE)

saveRDS(list(cs = cs_desnut, es = es_desnut, att = att_simples),
        "cs_desnut.rds")
message("✓ cs_desnut.rds salvo")

# ── 5. Descritivas para a tabela ──────────────────────────────
desc <- painel_did |>
  filter(n_acompanhamentos >= 30, ano == 2015) |>
  mutate(grupo = if_else(ano_tratamento == 0,
                         "Nunca abriu creche nova",
                         "Abriu creche nova")) |>
  group_by(grupo) |>
  summarise(
    n_munis         = n_distinct(id_muni),
    pct_desnut_2015 = round(mean(pct_desnutricao, na.rm = TRUE), 2),
    n_acomp_median  = round(median(n_acompanhamentos, na.rm = TRUE)),
    .groups = "drop"
  )

total_acomp <- sum(painel_did$n_acompanhamentos[painel_did$n_acompanhamentos >= 30],
                   na.rm = TRUE)

saveRDS(list(desc = desc, total_acomp = total_acomp), "desc_artigo.rds")
message("✓ desc_artigo.rds salvo")
message("\nPronto! Agora rode: quarto render index.qmd")

