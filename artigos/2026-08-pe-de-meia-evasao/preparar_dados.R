# ============================================================================
# preparar_dados.R  -  artigo "Pe-de-Meia e evasao no ensino medio"
#
# Roda UMA VEZ, localmente, antes de `quarto render index.qmd`.
# Gera / atualiza todos os arquivos que o index.qmd le da pasta data/:
#
#   data/freq_liquida_7141.csv          (SIDRA - Tabela 7141)
#   data/motivo_nao_freq_7220.csv       (SIDRA - Tabela 7220)
#   data/composicao_renda_15_17.csv     (microdados PNADC - script de renda)
#   data/estimativa_renda_detalhada_15_17.csv
#   data/mediana_renda_motivo_15_17.csv
#   data/variacao_renda_15_17.csv
#   data/estimativa_faixa_etaria_2024_2025.csv
#
# O index.qmd NAO baixa microdados nem chama a API do SIDRA - ele so le esses
# CSVs ja tratados. Toda a parte pesada (download da PNADC via PNADcIBGE,
# desenho amostral, survey_total) fica nos scripts do repositorio
# exercicios-pnad e e apenas COPIADA para ca por este script.
#
# Uso:
#   setwd("artigos/2026-08-pe-de-meia-evasao")
#   source("preparar_dados.R")
# ============================================================================

# --- Garante que o working directory seja a pasta DESTE script -------------
# (assim data/ sempre cai em artigos/2026-08-pe-de-meia-evasao/data/, não
#  importa de onde o script foi chamado - RStudio "Source", Rscript, etc.)
.this <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
if (is.na(.this)) {
  .a <- commandArgs(FALSE)
  .this <- sub("^--file=", "", .a[grep("^--file=", .a)])
}
if (length(.this) == 1 && !is.na(.this) && nzchar(.this)) setwd(dirname(.this))
message("Working directory: ", getwd())

dir.create("data", showWarnings = FALSE)

# ----------------------------------------------------------------------------
# PARTE A - SIDRA (leve, via API). Reproduz as consultas de Pnad-Educacao.R.
# ----------------------------------------------------------------------------
if (!requireNamespace("sidrar", quietly = TRUE)) install.packages("sidrar")
library(sidrar)
suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
})

message("SIDRA 7141 - taxa ajustada de frequencia escolar liquida (15-17, EM)...")
freq_liquida <- get_sidra(
  api = "/t/7141/n1/all/n2/all/v/10282/p/all/c2/6794/c871/47818"
) |>
  rename(localidade = `Brasil e Grande Região`, ano = Ano, taxa = Valor) |>
  transmute(
    localidade = str_trim(localidade),
    ano        = as.integer(ano),
    taxa       = as.numeric(taxa)
  ) |>
  filter(!is.na(taxa)) |>
  arrange(localidade, ano)

write.csv(freq_liquida, "data/freq_liquida_7141.csv", row.names = FALSE)
message("  -> data/freq_liquida_7141.csv (", nrow(freq_liquida), " linhas)")

message("SIDRA 7220 - motivo de nao frequencia escolar (15 a 29 anos)...")
motivo_nao_freq <- get_sidra(
  api = "/t/7220/n1/all/n2/all/v/10406/p/all/c86/95251/c879/all"
) |>
  rename(
    localidade = `Brasil e Grande Região`,
    ano        = Ano,
    motivo     = `Principal motivo de atualmente não frequentar escola ou outro curso`,
    valor      = Valor
  ) |>
  transmute(
    localidade = str_trim(localidade),
    ano        = as.integer(ano),
    motivo     = as.character(motivo),
    valor      = as.numeric(valor)
  ) |>
  filter(motivo != "Total", !is.na(valor)) |>   # "Total" = soma dos motivos
  arrange(localidade, motivo, ano)

write.csv(motivo_nao_freq, "data/motivo_nao_freq_7220.csv", row.names = FALSE)
message("  -> data/motivo_nao_freq_7220.csv (", nrow(motivo_nao_freq), " linhas)")

# ----------------------------------------------------------------------------
# PARTE B - microdados PNADC (pesado). Apenas copia os CSVs ja gerados por:
#   exercicios-pnad/analise_renda_peedemeia_microdados.R   (composicao/renda)
#   exercicios-pnad/analise_peedemeia_microdados.R         (faixa etaria)
# Rode esses dois scripts la primeiro (eles baixam a PNADC e salvam em
# exercicios-pnad/resultados_peedemeia/). Ajuste o caminho se a pasta mudar.
# ----------------------------------------------------------------------------
# (a partir de artigos/2026-08-pe-de-meia-evasao/: sobe ate Acessados/ e desce em Pessoal/)
dir_micro <- "../../../../../Pessoal/exercicios-pnad/resultados_peedemeia"

arquivos_micro <- c(
  "composicao_renda_15_17.csv",
  "estimativa_renda_detalhada_15_17.csv",
  "mediana_renda_motivo_15_17.csv",
  "variacao_renda_15_17.csv",
  "estimativa_faixa_etaria_2024_2025.csv"
)

if (dir.exists(dir_micro)) {
  for (a in arquivos_micro) {
    orig <- file.path(dir_micro, a)
    if (file.exists(orig)) {
      file.copy(orig, file.path("data", a), overwrite = TRUE)
      message("  copiado: data/", a)
    } else {
      warning("nao encontrado (rode os scripts de microdados primeiro): ", orig)
    }
  }
} else {
  warning(
    "Pasta de resultados de microdados nao encontrada: ", dir_micro, "\n",
    "Os CSVs de microdados ja versionados em data/ continuam validos - ",
    "so rode esta parte se precisar regenera-los."
  )
}

message("\nPronto. Agora: quarto render index.qmd")
