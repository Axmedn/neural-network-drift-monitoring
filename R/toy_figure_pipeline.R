# Redraws the toy-simulation figure (embedding space + r-chart) from the
# cache written by R/toy_example.R. No model training is needed: it reads
# the cache file only and repeats the figure code with clearer legends and
# both panels in one figure.
#
# Run from the repository root after R/toy_example.R:
#   Rscript R/toy_figure_pipeline.R

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(ggplot2); library(patchwork)
})

args       <- commandArgs(trailingOnly = TRUE)
cache_file <- if (length(args) >= 1) args[1] else
  "toy_cache_FAST=FALSE_ntr=5000_nic=500_nooc=500_seed=42.rds"
out_file   <- if (length(args) >= 2) args[2] else "figures/toy_pipeline.png"
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

alpha      <- 0.05
plot_R     <- 2000
plot_delta <- 0.5

cache <- readRDS(cache_file)
M_tr <- cache$M_tr; y_tr <- cache$y_tr; pred_tr <- cache$pred_tr
prob_tr <- cache$prob_tr
M_ic <- cache$M_ic; pred_ic <- cache$pred_ic
M_ooc_list <- cache$M_ooc_list; pred_ooc_list <- cache$pred_ooc_list

# Reference pools, as in toy_example.R
correct        <- (pred_tr == y_tr)
use_confidence <- any(!correct)
conf_tr        <- abs(prob_tr - 0.5)
pools <- lapply(0:1, function(cls) {
  idx <- which(y_tr == cls & correct)
  if (use_confidence) idx <- idx[order(conf_tr[idx], decreasing = TRUE)]
  M_tr[idx, , drop = FALSE]
})

md_fit <- function(R, ridge = 1e-6)
  list(mu = colMeans(R), Sinv = solve(cov(R) + diag(ridge, ncol(R))))
md_depth <- function(Xq, p) {
  Z <- sweep(as.matrix(Xq), 2, p$mu, "-")
  1 / (1 + rowSums((Z %*% p$Sinv) * Z))
}
r_stat <- function(d_new, d_sorted)
  findInterval(d_new, d_sorted, rightmost.closed = TRUE) / length(d_sorted)

ref    <- lapply(pools, function(p) p[seq_len(plot_R), , drop = FALSE])
md_p   <- lapply(ref, md_fit)
ref_ds <- lapply(seq_along(ref), function(j) sort(md_depth(ref[[j]], md_p[[j]])))

r_md <- function(M, chat) {
  r <- rep(NA_real_, nrow(M))
  for (cls in 0:1) {
    idx <- which(chat == cls); if (!length(idx)) next
    r[idx] <- r_stat(md_depth(M[idx, , drop = FALSE], md_p[[cls + 1]]),
                     ref_ds[[cls + 1]])
  }
  r
}

nm      <- sprintf("delta=%.1f", plot_delta)
M_ooc   <- M_ooc_list[[nm]]
r_ic    <- r_md(M_ic, pred_ic)
r_ooc   <- r_md(M_ooc, pred_ooc_list[[nm]])

# Colours
col_ref0 <- "#2C6E9E"; col_ref1 <- "#3A9A8A"
col_ic   <- "#9AA5B1"; col_ooc  <- "#C8453B"; col_sig <- "#5B1A14"

# (a) embedding space

lab <- c(ref0 = "Reference set, class 0", ref1 = "Reference set, class 1",
         ic = "In-control test", ooc = "Out-of-control")
sc <- bind_rows(
  tibble(e1 = ref[[1]][, 1], e2 = ref[[1]][, 2], g = "ref0"),
  tibble(e1 = ref[[2]][, 1], e2 = ref[[2]][, 2], g = "ref1"),
  tibble(e1 = M_ic[, 1],     e2 = M_ic[, 2],     g = "ic"),
  tibble(e1 = M_ooc[, 1],    e2 = M_ooc[, 2],    g = "ooc")
) |> mutate(g = factor(g, levels = names(lab)))

base_theme <- theme_classic(base_size = 11) + theme(
  legend.position = "bottom", legend.title = element_blank(),
  plot.tag = element_text(face = "bold"),
  panel.border = element_rect(colour = "grey70", fill = NA, linewidth = 0.4))

p_embed <- ggplot() +
  geom_point(data = sc, aes(e1, e2, colour = g, shape = g),
             size = 1.3, alpha = 0.75) +
  scale_colour_manual(values = c(ref0 = col_ref0, ref1 = col_ref1,
                                 ic = col_ic, ooc = col_ooc), labels = lab) +
  scale_shape_manual(values = c(ref0 = 16, ref1 = 17, ic = 15, ooc = 3),
                     labels = lab) +
  guides(colour = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
  labs(x = expression(m[list(t, 1)]), y = expression(m[list(t, 2)])) +
  base_theme

# (b) r-chart
n_ic <- length(r_ic)
cd <- bind_rows(
  tibble(t = seq_len(n_ic), r = r_ic, grp = "In-control"),
  tibble(t = n_ic + seq_along(r_ooc), r = r_ooc, grp = "Out-of-control")
) |> mutate(signal = r <= alpha)

p_chart <- ggplot(cd, aes(t, r)) +
  annotate("rect", xmin = n_ic + 0.5, xmax = nrow(cd) + 0.5,
           ymin = -Inf, ymax = Inf, fill = "#F6DCD8", alpha = 0.6) +
  geom_hline(yintercept = alpha, linetype = "dashed", linewidth = 0.5,
             colour = col_sig) +
  geom_point(data = filter(cd, !signal), aes(colour = grp),
             size = 0.9, alpha = 0.8) +
  geom_point(data = filter(cd, signal), colour = col_sig,
             shape = 4, size = 1.4, stroke = 0.6) +
  scale_colour_manual(values = c("In-control" = col_ic,
                                 "Out-of-control" = col_ooc)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(x = "Observation index t", y = expression(r(bold(m)[t]))) +
  base_theme + theme(legend.position = "none")

fig <- (p_embed / p_chart) + plot_layout(heights = c(1.2, 1)) +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")

ggsave(out_file, fig, width = 7, height = 7.4, dpi = 300, bg = "white")

# Numbers quoted in the caption and text
cat(sprintf("IC false alarms: %d of %d (%.3f)\n",
            sum(r_ic <= alpha), n_ic, mean(r_ic <= alpha)))
cat(sprintf("OOC r exactly 0: %d of %d; max r = %.4f\n", sum(r_ooc == 0), length(r_ooc), max(r_ooc)))
cat(sprintf("OOC signals: %d of %d (%.3f)\n",
            sum(r_ooc <= alpha), length(r_ooc), mean(r_ooc <= alpha)))
