# =============================================================================
# Perfil dos beneficiários do Bolsa Família
# PNAD Contínua – 1ª visita: 2015 e 2025  (análises separadas)
# Pacotes: PNADcIBGE + srvyr
# =============================================================================
# Nota: get_pnadc() com interview= retorna svyrep.design com pesos bootstrap
# (V1032). Nunca reconstruir o plano manualmente. Análises por domínio via
# subset() sobre o objeto de desenho.
# =============================================================================

library(PNADcIBGE)
library(srvyr)       # wrapper dplyr sobre o pacote survey
library(tidyverse)

# -----------------------------------------------------------------------------
# 1. Variáveis de interesse
# -----------------------------------------------------------------------------

vars <- c(
  "V5002A",   # Recebeu Bolsa Família / Auxílio Brasil  (1=Sim, 2=Não)
  "V5002A2",  # Valor recebido (R$)
  "V5001A",   # Recebeu BPC-LOAS
  "V5003A",   # Recebeu outros programas sociais
  "V2007",    # Sexo
  "V2009",    # Idade
  "V2010",    # Cor ou raça
  "V2001",    # Nº pessoas no domicílio
  "VD3004",   # Nível de instrução
  "VD4001",   # Condição na força de trabalho
  "VD4002",   # Condição de ocupação
  "VD4019"    # Rendimento habitual de todos os trabalhos
)

# -----------------------------------------------------------------------------
# 2. Download — design=TRUE (padrão) mantém o plano amostral bootstrap
# -----------------------------------------------------------------------------

pnadc_2015 <- get_pnadc(year = 2015, 
                        interview = 1, 
                        vars = vars, 
                        labels = TRUE, 
                        design = TRUE)

pnadc_2025 <- get_pnadc(year = 2025, 
                        interview = 1, 
                        vars = vars, 
                        labels = TRUE, 
                        design = TRUE)

pnadc_2015.trat = as_survey(pnadc_2015)
pnadc_2025.trat = as_survey(pnadc_2025)

# -----------------------------------------------------------------------------
# 3. Variáveis derivadas (mutate sobre o objeto srvyr)
# -----------------------------------------------------------------------------

adicionar_vars <- function(svy) {
  svy |>
    mutate(
      # Recebe BF (lógico)
      bf = V5002A == "Sim",

      # Sexo harmonizado (2015 usa Masculino/Feminino; 2025 usa Homem/Mulher)
      sexo = case_when(
        as.character(V2007) %in% c("Homem",   "Masculino") ~ "Homem",
        as.character(V2007) %in% c("Mulher",  "Feminino")  ~ "Mulher"
      ),

      # Faixa etária
      faixa_etaria = case_when(
        V2009 <  18  ~ "0-17",
        V2009 <= 29  ~ "18-29",
        V2009 <= 44  ~ "30-44",
        V2009 <= 59  ~ "45-59",
        V2009 >= 60  ~ "60+"
      ) |> factor(levels = c("0-17","18-29","30-44","45-59","60+")),

      # Raça (Ignorado → NA)
      raca = if_else(V2010 == "Ignorado", NA_character_, as.character(V2010)),

      # Instrução (agregada)
      instrucao = case_when(
        VD3004 %in% c("Sem instrução e menos de 1 ano de estudo",
                      "Fundamental incompleto ou equivalente")    ~ "Fund. incompleto ou menos",
        VD3004 == "Fundamental completo ou equivalente"           ~ "Fund. completo",
        VD3004 %in% c("Médio incompleto ou equivalente",
                      "Médio completo ou equivalente")            ~ "Médio",
        VD3004 %in% c("Superior incompleto ou equivalente",
                      "Superior completo")                        ~ "Superior"
      ) |> factor(levels = c("Fund. incompleto ou menos","Fund. completo",
                             "Médio","Superior")),

      # Situação no mercado de trabalho
      sit_trab = case_when(
        as.character(VD4002) == "Pessoas ocupadas"    ~ "Ocupado",
        as.character(VD4002) == "Pessoas desocupadas" ~ "Desocupado",
        as.character(VD4001) == "Pessoas fora da força de trabalho" ~ "Fora da FT"
      ),

      # Tamanho do domicílio
      tam_dom = case_when(
        V2001 == 1        ~ "1 pessoa",
        V2001 == 2        ~ "2 pessoas",
        V2001 %in% c(3,4) ~ "3-4 pessoas",
        V2001 >= 5        ~ "5 ou mais"
      ) |> factor(levels = c("1 pessoa","2 pessoas","3-4 pessoas","5 ou mais"))
    )
}

pnadc_2015.trat <- adicionar_vars(pnadc_2015.trat)
pnadc_2025.trat <- adicionar_vars(pnadc_2025.trat)

# -----------------------------------------------------------------------------
# 4. Subconjunto: apenas beneficiários do BF
#    filter() sobre objeto srvyr preserva o plano bootstrap
# -----------------------------------------------------------------------------

bf_2015 <- pnadc_2015.trat |> filter(bf == TRUE)
bf_2025 <- pnadc_2025.trat |> filter(bf == TRUE)

# -----------------------------------------------------------------------------
# 5. Tabulações — composição interna dos beneficiários
# -----------------------------------------------------------------------------

composicao <- function(svy_bf, var, rotulo) {
  svy_bf |>
    filter(!is.na(.data[[var]])) |>
    group_by(.data[[var]]) |>
    summarise(
      n_est = survey_total(vartype = "ci"),
      prop  = survey_prop(vartype  = "ci")
    ) |>
    rename(categoria = 1) |>
    mutate(variavel = rotulo)
}

# Taxa de cobertura do BF dentro de cada grupo
cobertura <- function(svy_full, var, rotulo) {
  svy_full |>
    filter(!is.na(.data[[var]]), !is.na(bf)) |>
    group_by(.data[[var]]) |>
    summarise(taxa_bf = survey_mean(bf, vartype = "ci")) |>
    rename(categoria = 1) |>
    mutate(variavel = rotulo)
}

vars_perfil <- list(
  sexo         = "sexo",
  faixa_etaria = "faixa_etaria",
  raca         = "raca",
  instrucao    = "instrucao",
  sit_trab     = "sit_trab",
  tam_dom      = "tam_dom"
)

perfil_2015 <- map2_dfr(vars_perfil, names(vars_perfil),
                        ~ composicao(bf_2015, .x, .y)) |> mutate(ano = 2015)

perfil_2025 <- map2_dfr(vars_perfil, names(vars_perfil),
                        ~ composicao(bf_2025, .x, .y)) |> mutate(ano = 2025)

cob_2015 <- map2_dfr(vars_perfil, names(vars_perfil),
                     ~ cobertura(pnadc_2015.trat, .x, .y)) |> mutate(ano = 2015)

cob_2025 <- map2_dfr(vars_perfil, names(vars_perfil),
                     ~ cobertura(pnadc_2025.trat, .x, .y)) |> mutate(ano = 2025)

# -----------------------------------------------------------------------------
# 6. Totais e renda
# -----------------------------------------------------------------------------

n_bf <- bind_rows(
  bf_2015 |> summarise(n_est = survey_total(vartype = "ci")) |> mutate(ano = 2015),
  bf_2025 |> summarise(n_est = survey_total(vartype = "ci")) |> mutate(ano = 2025)
)

renda_bf <- bind_rows(
  bf_2015 |>
    summarise(
      renda_media    = survey_mean(VD4019, na.rm = TRUE, vartype = "ci"),
      renda_mediana  = survey_median(VD4019, na.rm = TRUE, vartype = "ci"),
      valor_bf_medio = survey_mean(as.numeric(as.character(V5002A2)),
                                   na.rm = TRUE, vartype = "ci")
    ) |> mutate(ano = 2015),
  bf_2025 |>
    summarise(
      renda_media    = survey_mean(VD4019, na.rm = TRUE, vartype = "ci"),
      renda_mediana  = survey_median(VD4019, na.rm = TRUE, vartype = "ci"),
      valor_bf_medio = survey_mean(as.numeric(as.character(V5002A2)),
                                   na.rm = TRUE, vartype = "ci")
    ) |> mutate(ano = 2025)
)

# -----------------------------------------------------------------------------
# 7. Visualizações
# -----------------------------------------------------------------------------

perfil_total <- bind_rows(perfil_2015, perfil_2025) |> mutate(ano = factor(ano))
cob_total    <- bind_rows(cob_2015,    cob_2025)    |> mutate(ano = factor(ano))

# Paleta e tema base
cores_ano <- c("2015" = "#9e2d4a", "2025" = "#c5b89f")
caption_padrao <- "Fonte: PNAD Contínua – 1ª visita (2015 e 2025), IBGE.\nNota: Barras de erro representam intervalos de confiança de 95%."

library(showtext)
showtext_auto()
font_add_google("Lato", "Lato")

tema_bf <- function() {
  theme_minimal(base_size = 14, base_family = "Lato") +
    theme(
      # Títulos
      plot.title    = element_text(face = "bold", size = 14, hjust = 0, color = "#1a1a2e"),
      plot.subtitle = element_text(size = 11, hjust = 0, color = "#444466",
                                   margin = margin(t = 3, b = 10)),
      plot.caption  = element_text(size = 8, color = "#777777", hjust = 0,
                                   margin = margin(t = 10)),
      # Eixos
      axis.title.y  = element_text(size = 10, color = "#555555"),
      axis.text     = element_text(size = 10, color = "#333333"),
      axis.ticks    = element_blank(),
      # Grid
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
      # Legenda
      legend.position    = "top",
      legend.title       = element_text(size = 10, face = "bold"),
      legend.text        = element_text(size = 10),
      legend.key.size    = unit(0.45, "cm"),
      # Margens do painel
      plot.margin = margin(12, 16, 10, 12)
    )
}

# Função de composição interna
grafico_comp <- function(df, var_nome, titulo, subtitulo,
                         ordenar = FALSE, flip = FALSE, nivel_order = NULL) {
  d <- df |> filter(variavel == var_nome, !is.na(categoria))
  if (!is.null(nivel_order)) {
    d <- d |> mutate(categoria = factor(categoria, levels = nivel_order))
  } else if (ordenar) {
    ord <- d |> filter(ano == "2025") |> arrange(desc(prop)) |>
      pull(categoria) |> as.character()
    d <- d |> mutate(categoria = factor(categoria, levels = ord))
  }
  p <- ggplot(d, aes(x = categoria, y = prop, fill = ano)) +
    geom_col(position = position_dodge(0.65), width = 0.6,
             color = "white", linewidth = 0.3) +
    geom_errorbar(aes(ymin = prop_low, ymax = prop_upp),
                  position = position_dodge(0.65), width = 0.18,
                  linewidth = 0.45, color = "#444444") +
    geom_text(aes(label = scales::percent(prop, accuracy = 0.1), y = prop_upp),
              position = position_dodge(0.65), vjust = -0.4,
              size = 2.8, color = "#333333") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, 0.12))) +
    scale_fill_manual(values = cores_ano, name = NULL,
                      labels = c("2015" = "2015 (1ª visita)",
                                 "2025" = "2025 (1ª visita)")) +
    labs(
      title    = titulo,
      subtitle = subtitulo,
      x = NULL,
      y = "Proporção dos beneficiários (%)",
      caption  = caption_padrao
    ) +
    tema_bf()

  if (flip) {
    p + coord_flip() +
      theme(panel.grid.major.x = element_line(color = "#e8e8e8", linewidth = 0.4),
            panel.grid.major.y = element_blank())
  } else {
    p + theme(axis.text.x = element_text(angle = 0))
  }
}

# Função de cobertura
grafico_cob <- function(df, var_nome, titulo, subtitulo, flip = FALSE) {
  d <- df |> filter(variavel == var_nome, !is.na(categoria))
  p <- ggplot(d, aes(x = reorder(categoria, taxa_bf), y = taxa_bf, fill = ano)) +
    geom_col(position = position_dodge(0.65), width = 0.6,
             color = "white", linewidth = 0.3) +
    geom_errorbar(aes(ymin = taxa_bf_low, ymax = taxa_bf_upp),
                  position = position_dodge(0.65), width = 0.18,
                  linewidth = 0.45, color = "#444444") +
    geom_text(aes(label = scales::percent(taxa_bf, accuracy = 0.1), y = taxa_bf_upp),
              position = position_dodge(0.65), vjust = -0.4,
              size = 2.8, color = "#333333") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, 0.12))) +
    scale_fill_manual(values = cores_ano, name = NULL,
                      labels = c("2015" = "2015 (1ª visita)",
                                 "2025" = "2025 (1ª visita)")) +
    labs(
      title    = titulo,
      subtitle = subtitulo,
      x = NULL,
      y = "% do grupo que recebe BF",
      caption  = caption_padrao
    ) +
    tema_bf()

  if (flip) {
    p + coord_flip() +
      theme(panel.grid.major.x = element_line(color = "#e8e8e8", linewidth = 0.4),
            panel.grid.major.y = element_blank())
  } else {
    p + theme(axis.text.x = element_text(angle = 0))
  }
}

# ── Gráficos de composição ────────────────────────────────────────────────────

grafico_comp(
  perfil_total, "sexo",
  titulo    = "Composição dos beneficiários do Bolsa Família por sexo",
  subtitulo = "Distribuição percentual entre homens e mulheres que recebem o benefício"
)

grafico_comp(
  perfil_total, "faixa_etaria",
  titulo    = "Composição dos beneficiários do Bolsa Família por faixa etária",
  subtitulo = "Distribuição percentual por grupos de idade"
)

grafico_comp(
  perfil_total, "raca",
  titulo    = "Composição dos beneficiários do Bolsa Família por cor/raça",
  subtitulo = "Ordenado pela proporção em 2025",
  ordenar   = TRUE
)

grafico_comp(
  perfil_total, "instrucao",
  titulo      = "Composição dos beneficiários do Bolsa Família por nível de instrução",
  subtitulo   = "Do menor ao maior nível educacional",
  flip        = TRUE,
  nivel_order = c("Superior", "Médio", "Fund. completo", "Fund. incompleto ou menos")
)

grafico_comp(
  perfil_total, "sit_trab",
  titulo    = "Composição dos beneficiários por situação no mercado de trabalho",
  subtitulo = "Ocupados, desocupados e fora da força de trabalho"
)

grafico_comp(
  perfil_total, "tam_dom",
  titulo    = "Composição dos beneficiários do Bolsa Família por tamanho do domicílio",
  subtitulo = "Número de moradores no domicílio"
)

# ── Gráficos de cobertura ─────────────────────────────────────────────────────

grafico_cob(
  cob_total, "raca",
  titulo    = "Taxa de cobertura do Bolsa Família por cor/raça",
  subtitulo = "Percentual que recebe o benefício dentro de cada grupo racial",
  flip      = TRUE
)

grafico_cob(
  cob_total, "instrucao",
  titulo    = "Taxa de cobertura do Bolsa Família por nível de instrução",
  subtitulo = "Percentual que recebe o benefício dentro de cada nível educacional",
  flip      = TRUE
)

grafico_cob(
  cob_total, "sit_trab",
  titulo    = "Taxa de cobertura do Bolsa Família por situação no trabalho",
  subtitulo = "Percentual que recebe o benefício dentro de cada categoria"
)

# -----------------------------------------------------------------------------
# 8. Salvar dados processados para uso no artigo (sem os microdados)
# -----------------------------------------------------------------------------

dados_bf <- list(
  perfil    = perfil_total,
  cobertura = cob_total,
  totais    = n_bf,
  renda     = renda_bf
)

saveRDS(dados_bf, file = "dados_bf.rds")

# -----------------------------------------------------------------------------

