# ============================================================
#  LaTeX Table Generator — corrected
#  Run from: improvements/
#  Reads from: toy_simulation_outputs/ and mnist_experiment_outputs/
#  Writes to:  latex_tables/
# ============================================================

library(dplyr); library(tidyr); library(readr)

toy_dir   <- "toy_simulation_outputs"
mnist_dir <- "mnist_experiment_outputs"
tex_dir   <- "latex_tables"
dir.create(tex_dir, showWarnings = FALSE)

write_tex <- function(lines, filename) {
  writeLines(lines, file.path(tex_dir, filename))
  cat("Written:", file.path(tex_dir, filename), "\n")
}

# ============================================================
#  TOY TABLE 1 — FAR and SR
# ============================================================
toy1 <- read_csv(file.path(toy_dir, "toy_table1_FAR_SR.csv"),
                 show_col_types = FALSE)

toy1_wide <- toy1 |>
  pivot_wider(names_from  = R_size,
              values_from = c(FAR, SR),
              names_glue  = "{.value}_{R_size}")

lines <- c(
  "% TOY Table 1 — FAR and SR",
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Toy simulation (IC calibration): false-alarm rate (FAR)",
  "and in-control signal rate (SR) for each depth method and reference",
  "size $|R_c|$ (per class). Nominal level $\\alpha=0.05$.}",
  "\\label{tab:toy-ic-calibration}",
  "\\small",
  "\\sisetup{round-mode=places,round-precision=3}",
  "\\begin{tabular}{l",
  "  S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]",
  "  S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]}",
  "\\toprule",
  "& \\multicolumn{3}{c}{FAR} & \\multicolumn{3}{c}{SR} \\\\",
  "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
  "Method & {$|R_c|{=}2000$} & {$|R_c|{=}3000$} & {$|R_c|{=}4000$}",
  "       & {$|R_c|{=}2000$} & {$|R_c|{=}3000$} & {$|R_c|{=}4000$} \\\\",
  "\\midrule"
)

for (i in seq_len(nrow(toy1_wide))) {
  r <- toy1_wide[i, ]
  lines <- c(lines, sprintf(
    "  %s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\",
    r$method,
    r$FAR_2000, r$FAR_3000, r$FAR_4000,
    r$SR_2000,  r$SR_3000,  r$SR_4000))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
write_tex(lines, "toy_table1_FAR_SR.tex")

# ============================================================
#  TOY TABLE 2 — CDR single run
# ============================================================
toy2 <- read_csv(file.path(toy_dir, "toy_table2_CDR_singlerun.csv"),
                 show_col_types = FALSE)

delta_cols <- setdiff(names(toy2), "method")

lines <- c(
  "% TOY Table 2 — CDR single run",
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Toy simulation (OOC detection): concept-drift detection",
  "rate (CDR) at $|R_c|=4000$ across drift levels $\\delta$.",
  "At $\\delta=0$ the OOC distribution coincides with the class-0",
  "IC distribution (sanity check). Nominal level $\\alpha=0.05$;",
  "$n_{\\mathrm{OOC}}=500$ per drift level.}",
  "\\label{tab:toy-cdr}",
  "\\small",
  "\\sisetup{round-mode=places,round-precision=3}",
  "\\begin{tabular}{l",
  "  S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]",
  "  S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]}",
  "\\toprule",
  "& \\multicolumn{6}{c}{CDR at drift level $\\delta$} \\\\",
  "\\cmidrule(lr){2-7}",
  "Method & {0.0} & {0.1} & {0.2} & {0.3} & {0.4} & {0.5} \\\\",
  "\\midrule"
)

for (i in seq_len(nrow(toy2))) {
  r    <- toy2[i, ]
  vals <- sapply(delta_cols, function(cn) sprintf("%.3f", r[[cn]]))
  lines <- c(lines,
             paste0("  ", r$method, " & ",
                    paste(vals, collapse = " & "), " \\\\"))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
write_tex(lines, "toy_table2_CDR_singlerun.tex")

# ============================================================
#  TOY TABLE 3 — Sensitivity at R = 4000
#  Uses plain l columns with $\pm$ — avoids siunitx parsing issues
# ============================================================
toy3 <- read_csv(
  file.path(toy_dir, "toy_table3_sensitivity_R4000.csv"),
  show_col_types = FALSE) |>
  filter(method != "SD")   # SD excluded — near-zero CDR throughout

delta_cols3 <- setdiff(names(toy3), "method")

# Format each cell as "mean ± SD" string
toy3_fmt <- toy3
for (cn in delta_cols3) {
  toy3_fmt[[cn]] <- sprintf("$%s$", toy3[[cn]])
}

lines <- c(
  "% TOY Table 3 — Sensitivity R=4000",
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Toy simulation (OOC sensitivity): CDR mean $\\pm$ SD",
  "across 10 independent OOC resamples at $|R_c|=4000$.",
  "SD is excluded as its CDR is near zero throughout.}",
  "\\label{tab:toy-sensitivity}",
  "\\small",
  paste0("\\begin{tabular}{l ",
         paste(rep("l", length(delta_cols3)), collapse = " "), "}"),
  "\\toprule",
  paste0("Method & ",
         paste(sprintf("{$\\delta=%s$}", delta_cols3),
               collapse = " & "), " \\\\"),
  "\\midrule"
)

for (i in seq_len(nrow(toy3_fmt))) {
  r    <- toy3_fmt[i, ]
  vals <- sapply(delta_cols3, function(cn) r[[cn]])
  lines <- c(lines,
             paste0("  ", r$method, " & ",
                    paste(vals, collapse = " & "), " \\\\"))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
write_tex(lines, "toy_table3_sensitivity_R4000.tex")

# ============================================================
#  MNIST TABLE 1 — FAR, SR, SR_M, SR_C
#  HDr rows at R=2000 and R=3000 removed (under-calibrated)
# ============================================================
m1 <- read_csv(file.path(mnist_dir, "mnist_table1_FAR_SR.csv"),
               show_col_types = FALSE) |>
  filter(!(method == "HDr" & R_size %in% c(2000, 3000)))

lines <- c(
  "% MNIST Table 1 — FAR and SR",
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{MNIST (IC calibration and Phase~II IC signal rates):",
  "FAR is the Phase~I false-alarm rate at nominal level $\\alpha=0.05$.",
  "SR is the class-weighted IC signal rate on the test set.",
  "$\\mathrm{SR}_M$ and $\\mathrm{SR}_C$ are the signal rates among",
  "misclassified and correctly classified test samples respectively.",
  "HDr is reported only at $|R_c|=4000$ due to under-calibration",
  "at smaller reference sizes (FAR $=0.000$ and $0.005$",
  "at $|R_c|=2000$ and $3000$ respectively).}",
  "\\label{tab:mnist-far-sr}",
  "\\small",
  "\\sisetup{round-mode=places,round-precision=3}",
  "\\begin{tabular}{l c",
  "  S[table-format=1.3] S[table-format=1.3]",
  "  S[table-format=1.3] S[table-format=1.3]}",
  "\\toprule",
  "Method & $|R_c|$ & {FAR} & {SR}",
  "  & {$\\mathrm{SR}_M$} & {$\\mathrm{SR}_C$} \\\\",
  "\\midrule"
)

prev_method <- ""
for (i in seq_len(nrow(m1))) {
  r <- m1[i, ]
  if (r$method != prev_method && i > 1)
    lines <- c(lines, "\\addlinespace")
  lines <- c(lines, sprintf(
    "  %s & %d & %.3f & %.3f & %.3f & %.3f \\\\",
    r$method, r$R_size, r$FAR, r$SR, r$SR_M, r$SR_C))
  prev_method <- r$method
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
write_tex(lines, "mnist_table1_FAR_SR.tex")

# ============================================================
#  MNIST TABLE 2 — CDR single run
#  Fix column headers: Gaussian_s0.1 -> $\sigma=0.1$
# ============================================================
m2 <- read_csv(
  file.path(mnist_dir, "mnist_table2_CDR_singlerun.csv"),
  show_col_types = FALSE)

cond_cols <- setdiff(names(m2), "method")

# Map raw column names to formatted LaTeX headers
header_map <- c(
  "Gaussian_s0.1" = "$\\sigma=0.1$",
  "Gaussian_s0.2" = "$\\sigma=0.2$",
  "Gaussian_s0.3" = "$\\sigma=0.3$",
  "Gaussian_s0.4" = "$\\sigma=0.4$",
  "FashionMNIST"  = "FashionMNIST"
)
headers <- sapply(cond_cols, function(cn)
  ifelse(cn %in% names(header_map), header_map[[cn]], cn))

lines <- c(
  "% MNIST Table 2 — CDR single run",
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{MNIST (Phase~II OOC detection at $|R_c|=4000$):",
  "class-weighted CDR under additive Gaussian pixel noise",
  "($\\sigma \\in \\{0.1, 0.2, 0.3, 0.4\\}$) and FashionMNIST",
  "dataset shift. OOC accuracy and mean softmax confidence",
  "are reported in the prose as diagnostics of drift severity only.}",
  "\\label{tab:mnist-cdr}",
  "\\small",
  "\\sisetup{round-mode=places,round-precision=3}",
  paste0("\\begin{tabular}{l ",
         paste(rep("S[table-format=1.3]", length(cond_cols)),
               collapse = " "), "}"),
  "\\toprule",
  paste0("Method & ",
         paste(sprintf("{%s}", headers), collapse = " & "),
         " \\\\"),
  "\\midrule"
)

for (i in seq_len(nrow(m2))) {
  r    <- m2[i, ]
  vals <- sapply(cond_cols, function(cn)
    if (is.na(r[[cn]])) "{---}" else sprintf("%.3f", r[[cn]]))
  lines <- c(lines,
             paste0("  ", r$method, " & ",
                    paste(vals, collapse = " & "), " \\\\"))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
write_tex(lines, "mnist_table2_CDR_singlerun.tex")

# ============================================================
#  MNIST TABLE 3 — Sensitivity Gaussian at R = 4000
#  Uses plain l columns with $\pm$ strings — avoids siunitx issues
#  Label fixed to tab:mnist-sensitivity
# ============================================================
m_sens <- read_csv(
  file.path(mnist_dir, "mnist_sensitivity_full.csv"),
  show_col_types = FALSE) |>
  filter(R_size == 4000, grepl("^Gaussian", cond)) |>
  mutate(
    sigma_label = sprintf("$\\sigma=%.1f$", sigma),
    entry       = sprintf("$%.3f \\pm %.3f$", CDR_mean, CDR_sd)
  ) |>
  select(method, sigma_label, entry) |>
  pivot_wider(names_from = sigma_label, values_from = entry)

sigma_cols <- setdiff(names(m_sens), "method")

lines <- c(
  "% MNIST Table 3 — Gaussian sensitivity R=4000",
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{MNIST (OOC sensitivity at $|R_c|=4000$): CDR mean",
  "$\\pm$ SD across 10 independent OOC resamples under Gaussian",
  "pixel noise. HDr is excluded at smaller reference sizes due to",
  "under-calibration. The network, embeddings, and Phase~I reference",
  "sets are held fixed; only the OOC draw varies across resamples.}",
  "\\label{tab:mnist-sensitivity}",
  "\\small",
  paste0("\\begin{tabular}{l ",
         paste(rep("l", length(sigma_cols)), collapse = " "), "}"),
  "\\toprule",
  paste0("Method & ",
         paste(sprintf("{%s}", sigma_cols), collapse = " & "),
         " \\\\"),
  "\\midrule"
)

for (i in seq_len(nrow(m_sens))) {
  r    <- m_sens[i, ]
  vals <- sapply(sigma_cols, function(cn) r[[cn]])
  lines <- c(lines,
             paste0("  ", r$method, " & ",
                    paste(vals, collapse = " & "), " \\\\"))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
write_tex(lines, "mnist_table3_sensitivity_R4000.tex")

# ============================================================
#  MNIST TABLE 4 — FashionMNIST sensitivity at R = 4000
# ============================================================
m_fash <- read_csv(
  file.path(mnist_dir, "mnist_sensitivity_full.csv"),
  show_col_types = FALSE) |>
  filter(R_size == 4000, cond == "FashionMNIST") |>
  select(method, CDR_mean, CDR_sd, CI_lo, CI_hi)

lines <- c(
  "% MNIST Table 4 — FashionMNIST sensitivity R=4000",
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{MNIST (FashionMNIST sensitivity at $|R_c|=4000$):",
  "CDR mean, SD, and 95\\% bootstrap confidence interval",
  "across 10 independent OOC resamples.}",
  "\\label{tab:mnist-sensitivity-fashion}",
  "\\small",
  "\\sisetup{round-mode=places,round-precision=3}",
  "\\begin{tabular}{l",
  "  S[table-format=1.3] S[table-format=1.3]",
  "  S[table-format=1.3] S[table-format=1.3]}",
  "\\toprule",
  "Method & {CDR mean} & {SD}",
  "  & {95\\% CI lower} & {95\\% CI upper} \\\\",
  "\\midrule"
)

for (i in seq_len(nrow(m_fash))) {
  r <- m_fash[i, ]
  lines <- c(lines, sprintf(
    "  %s & %.3f & %.3f & %.3f & %.3f \\\\",
    r$method, r$CDR_mean, r$CDR_sd, r$CI_lo, r$CI_hi))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
write_tex(lines, "mnist_table4_sensitivity_fashion.tex")

cat("\n── All tables written to:", tex_dir, "──\n")
cat("Toy    : toy_table1_FAR_SR.tex\n")
cat("         toy_table2_CDR_singlerun.tex\n")
cat("         toy_table3_sensitivity_R4000.tex\n")
cat("MNIST  : mnist_table1_FAR_SR.tex\n")
cat("         mnist_table2_CDR_singlerun.tex\n")
cat("         mnist_table3_sensitivity_R4000.tex\n")
cat("         mnist_table4_sensitivity_fashion.tex\n")