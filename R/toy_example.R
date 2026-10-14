
#  Toy Simulation: Depth-Based ANN Monitoring

suppressPackageStartupMessages({
  library(keras3); library(tensorflow); library(ddalpha)
  library(dplyr); library(tibble); library(tidyr)
  library(ggplot2); library(patchwork)
})

try(tensorflow::tf$config$experimental$enable_op_determinism(), silent = TRUE)
SEED <- 42
set.seed(SEED); tensorflow::set_random_seed(SEED)

#  CONFIG
FAST_DEV     <- FALSE
alpha        <- 0.05
delta_levels <- c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
plot_R       <- 2000
plot_delta   <- 0.5
OOC_SEEDS    <- 1:10
N_BOOT       <- 1000

if (FAST_DEV) {
  R_sizes <- c(50, 100, 200); n_tr <- 400; n_ic <- 100; n_ooc <- 100
} else {
  R_sizes <- c(2000, 3000, 4000); n_tr <- 5000; n_ic <- 500; n_ooc <- 500
}

# OUTPUT FOLDER
out <- file.path(getwd(), "toy_simulation_outputs")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
cat("Toy outputs will be saved to:", out, "\n")

#CACHE PATHS
cache_file       <- file.path(getwd(), sprintf(
  "toy_cache_FAST=%s_ntr=%d_nic=%d_nooc=%d_seed=%d.rds",
  FAST_DEV, n_tr, n_ic, n_ooc, SEED))
model_file       <- paste0(cache_file, "_model.keras")
embed_model_file <- paste0(cache_file, "_embed.keras")

# ── DATA HELPERS
tridiag <- function(d, rho) {
  S <- diag(d)
  for (i in seq_len(d - 1)) { S[i, i+1] <- rho; S[i+1, i] <- rho }
  S
}

rmnorm <- function(n, mu, S) {
  L <- t(chol(S))
  t(mu + L %*% matrix(rnorm(length(mu) * n), length(mu)))
}

# ── DATA + TRAIN + EMBED (CACHED
if (file.exists(cache_file) &&
    file.exists(model_file) &&
    file.exists(embed_model_file)) {
  cache         <- readRDS(cache_file)
  X_tr          <- cache$X_tr;  y_tr          <- cache$y_tr
  X_ic          <- cache$X_ic;  y_ic          <- cache$y_ic
  prob_tr       <- cache$prob_tr; pred_tr      <- cache$pred_tr
  M_tr          <- cache$M_tr;  acc_tr         <- cache$acc_tr
  prob_ic       <- cache$prob_ic; pred_ic      <- cache$pred_ic
  M_ic          <- cache$M_ic;  acc_ic         <- cache$acc_ic
  M_ooc_list    <- cache$M_ooc_list
  pred_ooc_list <- cache$pred_ooc_list
  model         <- load_model(model_file)
  embed_model   <- load_model(embed_model_file)
  d    <- ncol(X_tr)
  mu0  <- rep(0, d); mu1 <- rep(10, d)
  Sig1 <- tridiag(d, 0.3)
  
} else {
  
  d    <- 4
  Sig1 <- tridiag(d,  0.3)
  Sig2 <- tridiag(d, -0.3)
  mu0  <- rep(0, d)
  mu1  <- rep(10, d)
  
  X_tr <- rbind(rmnorm(n_tr, mu0, Sig1), rmnorm(n_tr, mu1, Sig2))
  y_tr <- c(rep(0L, n_tr), rep(1L, n_tr))
  X_ic <- rbind(rmnorm(n_ic, mu0, Sig1), rmnorm(n_ic, mu1, Sig2))
  y_ic <- c(rep(0L, n_ic), rep(1L, n_ic))
  
  ooc_list <- setNames(
    lapply(delta_levels, function(dv)
      rmnorm(n_ooc, (1-dv)*mu0 + dv*mu1, Sig1)),
    sprintf("delta=%.1f", delta_levels))
  
  model <- keras_model_sequential() |>
    layer_dense(3, activation = "relu",   input_shape = d) |>
    layer_dense(2, activation = "linear", name = "embed")  |>
    layer_dense(1, activation = "sigmoid")
  model |> compile(optimizer_adam(0.01),
                   loss = "binary_crossentropy", metrics = "accuracy")
  model |> fit(X_tr, y_tr, epochs = 100, batch_size = 32, verbose = 0)
  
  embed_model <- keras_model(model$inputs,
                             get_layer(model, "embed")$output)
  
  prob_tr <- as.vector(predict(model, X_tr, verbose = 0))
  pred_tr <- as.integer(prob_tr >= 0.5)
  M_tr    <- predict(embed_model, X_tr, verbose = 0)
  acc_tr  <- mean(pred_tr == y_tr)
  
  prob_ic <- as.vector(predict(model, X_ic, verbose = 0))
  pred_ic <- as.integer(prob_ic >= 0.5)
  M_ic    <- predict(embed_model, X_ic, verbose = 0)
  acc_ic  <- mean(pred_ic == y_ic)
  
  M_ooc_list    <- lapply(ooc_list, function(X)
    predict(embed_model, X, verbose = 0))
  pred_ooc_list <- lapply(ooc_list, function(X)
    as.integer(as.vector(predict(model, X, verbose = 0)) >= 0.5))
  
  saveRDS(list(
    X_tr=X_tr, y_tr=y_tr, X_ic=X_ic, y_ic=y_ic, ooc_list=ooc_list,
    prob_tr=prob_tr, pred_tr=pred_tr, M_tr=M_tr, acc_tr=acc_tr,
    prob_ic=prob_ic, pred_ic=pred_ic, M_ic=M_ic, acc_ic=acc_ic,
    M_ooc_list=M_ooc_list, pred_ooc_list=pred_ooc_list
  ), cache_file)
  save_model(model,       model_file)
  save_model(embed_model, embed_model_file)
  cat("Cache saved.\n")
}

cat(sprintf("Train acc: %.1f%%  |  IC test acc: %.1f%%\n",
            100*acc_tr, 100*acc_ic))

# REFERENCE POOLS
correct        <- (pred_tr == y_tr)
use_confidence <- any(!correct)
conf_tr        <- abs(prob_tr - 0.5)

pools <- lapply(0:1, function(cls) {
  idx <- which(y_tr == cls & correct)
  if (use_confidence) idx <- idx[order(conf_tr[idx], decreasing = TRUE)]
  M_tr[idx, , drop = FALSE]
})

stopifnot("Pool too small" = min(sapply(pools, nrow)) >= max(R_sizes))
cat(sprintf("Reference pooling: confidence-ordering = %s\n",
            ifelse(use_confidence, "ON", "OFF")))

# DEPTH FUNCTIONS
md_fit <- function(R, ridge = 1e-6)
  list(mu = colMeans(R), Sinv = solve(cov(R) + diag(ridge, ncol(R))))

md_depth <- function(Xq, p) {
  Z <- sweep(as.matrix(Xq), 2, p$mu, "-")
  1 / (1 + rowSums((Z %*% p$Sinv) * Z))
}

depth_HD <- function(Xq, R)
  tryCatch(
    as.vector(ddalpha::depth.halfspace(as.matrix(Xq), as.matrix(R),
                                       exact = TRUE)),
    error = function(e) rep(NA_real_, nrow(as.matrix(Xq))))

depth_PD <- function(Xq, R, num.directions = 500, seed = SEED)
  tryCatch(
    as.vector(ddalpha::depth.projection(as.matrix(Xq), as.matrix(R),
                                        method = "random",
                                        num.directions = num.directions,
                                        seed = seed)),
    error = function(e) rep(NA_real_, nrow(as.matrix(Xq))))

depth_SD <- function(Xq, R)
  tryCatch(
    as.vector(ddalpha::depth.simplicial(as.matrix(Xq), as.matrix(R))),
    error = function(e) rep(NA_real_, nrow(as.matrix(Xq))))

depth_fun <- function(m) switch(m, HD=depth_HD, PD=depth_PD, SD=depth_SD)
methods   <- c("MD", "HD", "PD", "SD")

# r-STATISTIC HELPERS
r_stat <- function(d_new, d_sorted)
  findInterval(d_new, d_sorted, rightmost.closed = TRUE) / length(d_sorted)

far_class    <- function(d_ref) mean(r_stat(d_ref, sort(d_ref)) <= alpha)

weighted_rate <- function(sig, chat) {
  n <- length(sig)
  sum(vapply(sort(unique(chat)), function(cl) {
    idx <- chat == cl & !is.na(sig)
    if (!any(idx)) 0 else (sum(chat == cl) / n) * mean(sig[idx])
  }, numeric(1)))
}

compute_r <- function(m, M, chat, ref_sets, ref_ds, md_p,
                      pd_seed = SEED) {
  r <- rep(NA_real_, nrow(M))
  for (cls in 0:1) {
    idx <- which(chat == cls); if (!length(idx)) next
    d_new <- if (m == "MD") {
      md_depth(M[idx, , drop=FALSE], md_p[[cls+1]])
    } else if (m == "PD") {
      depth_PD(M[idx, , drop=FALSE], ref_sets[[cls+1]], seed = pd_seed)
    } else {
      as.vector(depth_fun(m)(M[idx, , drop=FALSE], ref_sets[[cls+1]]))
    }
    ok <- !is.na(d_new)
    if (any(ok)) r[idx[ok]] <- r_stat(d_new[ok], ref_ds[[m]][[cls+1]])
  }
  r
}

#  SINGLE-RUN EVALUATION
ic_correct <- (pred_ic == y_ic)

toy_results <- bind_rows(lapply(R_sizes, function(R) {
  cat(sprintf("\n|R_c| = %d\n", R))
  ref  <- lapply(pools, function(p) p[seq_len(R), , drop=FALSE])
  md_p <- lapply(ref, md_fit)
  
  ref_depths <- list(
    MD = lapply(seq_along(ref),
                function(j) md_depth(ref[[j]], md_p[[j]])),
    HD = lapply(ref, function(Rc) {
      v <- depth_HD(Rc, Rc); if (anyNA(v)) NULL else v }),
    PD = lapply(ref, function(Rc) {
      v <- depth_PD(Rc, Rc); if (anyNA(v)) NULL else v }),
    SD = lapply(ref, function(Rc) {
      v <- depth_SD(Rc, Rc); if (anyNA(v)) NULL else v })
  )
  ref_ds <- lapply(ref_depths, function(dl)
    lapply(dl, function(v) if (is.null(v)) NULL else sort(v)))
  
  FAR <- vapply(methods, function(m) {
    rd <- ref_depths[[m]]
    if (is.null(rd) || any(sapply(rd, is.null))) return(NA_real_)
    mean(vapply(rd, far_class, numeric(1)))
  }, numeric(1))
  cat(sprintf("  FAR: %s\n",
              paste(sprintf("%s=%.3f", methods, FAR), collapse="  ")))
  
  ic_rows <- bind_rows(lapply(methods, function(m) {
    if (is.na(FAR[m]))
      return(tibble(R_size=R, method=m, delta=NA_real_,
                    FAR=NA_real_, SR=NA_real_, SR_M=NA_real_,
                    SR_C=NA_real_, CDR=NA_real_, phase="IC"))
    r   <- compute_r(m, M_ic, pred_ic, ref, ref_ds, md_p)
    sig <- !is.na(r) & r <= alpha
    tibble(R_size=R, method=m, delta=NA_real_, FAR=FAR[m],
           SR   = weighted_rate(sig, pred_ic),
           SR_M = if (any(!ic_correct))
             mean(sig[!ic_correct], na.rm=TRUE) else NA_real_,
           SR_C = if (any( ic_correct))
             mean(sig[ ic_correct], na.rm=TRUE) else NA_real_,
           CDR=NA_real_, phase="IC")
  }))
  
  ooc_rows <- bind_rows(lapply(seq_along(delta_levels), function(di) {
    dv      <- delta_levels[di]
    nm      <- names(M_ooc_list)[di]
    chat_oc <- pred_ooc_list[[nm]]
    bind_rows(lapply(methods, function(m) {
      if (is.na(FAR[m]))
        return(tibble(R_size=R, method=m, delta=dv,
                      FAR=NA_real_, SR=NA_real_, SR_M=NA_real_,
                      SR_C=NA_real_, CDR=NA_real_, phase="OOC"))
      r   <- compute_r(m, M_ooc_list[[nm]], chat_oc, ref, ref_ds, md_p)
      sig <- !is.na(r) & r <= alpha
      tibble(R_size=R, method=m, delta=dv, FAR=FAR[m],
             SR=NA_real_, SR_M=NA_real_, SR_C=NA_real_,
             CDR=weighted_rate(sig, chat_oc), phase="OOC")
    }))
  }))
  
  bind_rows(ic_rows, ooc_rows)
}))

toy_res <- toy_results |>
  mutate(across(c(FAR, SR, SR_M, SR_C, CDR), \(x) round(x, 3)))

toy_res |> filter(phase=="IC") |>
  select(method, R_size, FAR, SR) |>
  arrange(method, R_size) |> print(n=Inf)

toy_res |> filter(phase=="OOC", R_size==max(R_sizes)) |>
  select(method, delta, CDR) |>
  pivot_wider(names_from=delta, values_from=CDR) |> print()

# ── SENSITIVITY ANALYSIS


toy_sens_results <- bind_rows(lapply(R_sizes, function(R) {
  cat(sprintf("\n  |R_c| = %d\n", R))
  ref_s  <- lapply(pools, function(p) p[seq_len(R), , drop=FALSE])
  md_p_s <- lapply(ref_s, md_fit)
  ref_depths_s <- list(
    MD = lapply(seq_along(ref_s),
                function(j) md_depth(ref_s[[j]], md_p_s[[j]])),
    HD = lapply(ref_s, function(Rc) {
      v <- depth_HD(Rc, Rc); if (anyNA(v)) NULL else v }),
    PD = lapply(ref_s, function(Rc) {
      v <- depth_PD(Rc, Rc); if (anyNA(v)) NULL else v }),
    SD = lapply(ref_s, function(Rc) {
      v <- depth_SD(Rc, Rc); if (anyNA(v)) NULL else v })
  )
  ref_ds_s <- lapply(ref_depths_s, function(dl)
    lapply(dl, function(v) if (is.null(v)) NULL else sort(v)))
  
  bind_rows(lapply(OOC_SEEDS, function(s) {
    cat(sprintf("    seed %d\n", s))
    set.seed(s)
    ooc_s <- setNames(
      lapply(delta_levels, function(dv)
        rmnorm(n_ooc, (1-dv)*mu0 + dv*mu1, Sig1)),
      sprintf("delta=%.1f", delta_levels))
    M_ooc_s    <- lapply(ooc_s, function(X)
      predict(embed_model, X, verbose = 0))
    pred_ooc_s <- lapply(ooc_s, function(X)
      as.integer(as.vector(predict(model, X, verbose = 0)) >= 0.5))
    
    bind_rows(lapply(seq_along(delta_levels), function(di) {
      dv    <- delta_levels[di]
      M_oc  <- M_ooc_s[[names(ooc_s)[di]]]
      ch_oc <- pred_ooc_s[[names(ooc_s)[di]]]
      bind_rows(lapply(methods, function(m) {
        r   <- compute_r(m, M_oc, ch_oc, ref_s, ref_ds_s,
                         md_p_s, pd_seed = s)
        sig <- !is.na(r) & r <= alpha
        tibble(R_size=R, seed=s, method=m, delta=dv,
               CDR=weighted_rate(sig, ch_oc))
      }))
    }))
  }))
}))

# ── BOOTSTRAP CI
boot_ci_lo <- function(x, B = N_BOOT, conf = 0.95) {
  if (all(is.na(x))) return(NA_real_)
  x  <- x[!is.na(x)]
  bm <- replicate(B, mean(sample(x, length(x), replace=TRUE)))
  round(quantile(bm, (1-conf)/2, names=FALSE), 3)
}
boot_ci_hi <- function(x, B = N_BOOT, conf = 0.95) {
  if (all(is.na(x))) return(NA_real_)
  x  <- x[!is.na(x)]
  bm <- replicate(B, mean(sample(x, length(x), replace=TRUE)))
  round(quantile(bm, 1-(1-conf)/2, names=FALSE), 3)
}

toy_sens_summary <- toy_sens_results |>
  group_by(R_size, method, delta) |>
  summarise(
    CDR_mean = round(mean(CDR, na.rm=TRUE), 3),
    CDR_sd   = round(sd(CDR,   na.rm=TRUE), 3),
    CI_lo    = boot_ci_lo(CDR),
    CI_hi    = boot_ci_hi(CDR),
    .groups  = "drop"
  ) |>
  mutate(CDR_report = sprintf("%.3f +/- %.3f", CDR_mean, CDR_sd))

for (R in R_sizes) {
  cat(sprintf("\n  |R_c| = %d\n", R))
  toy_sens_summary |> filter(R_size==R) |>
    select(method, delta, CDR_report) |>
    pivot_wider(names_from=delta, values_from=CDR_report) |> print()
}

# SAVE ALL OUTPUTS
toy_table1 <- toy_res |> filter(phase=="IC") |>
  select(method, R_size, FAR, SR) |> arrange(method, R_size)

toy_table2 <- toy_res |> filter(phase=="OOC", R_size==max(R_sizes)) |>
  select(method, delta, CDR) |>
  pivot_wider(names_from=delta, values_from=CDR)

write.csv(toy_table1,
          file.path(out, "toy_table1_FAR_SR.csv"),
          row.names=FALSE)
write.csv(toy_table2,
          file.path(out, "toy_table2_CDR_singlerun.csv"),
          row.names=FALSE)

for (R in R_sizes) {
  tbl <- toy_sens_summary |> filter(R_size==R) |>
    select(method, delta, CDR_report) |>
    pivot_wider(names_from=delta, values_from=CDR_report)
  write.csv(tbl,
            file.path(out, sprintf("toy_table3_sensitivity_R%d.csv", R)),
            row.names=FALSE)
}

write.csv(toy_sens_summary,
          file.path(out, "toy_sensitivity_full.csv"),
          row.names=FALSE)
saveRDS(toy_sens_results,
        file.path(out, "toy_sensitivity_raw.rds"))
saveRDS(toy_sens_summary,
        file.path(out, "toy_sensitivity_summary.rds"))

sink(file.path(out, "toy_latex_numbers.txt"))
print(as.data.frame(toy_table1))
cat("\nTABLE 2: CDR single run, |R_c| =", max(R_sizes), "\n")
print(as.data.frame(toy_table2))
cat("\nTABLE 3: Sensitivity, |R_c| =", max(R_sizes), "(mean +/- SD)\n")
toy_sens_summary |> filter(R_size==max(R_sizes)) |>
  select(method, delta, CDR_mean, CDR_sd, CI_lo, CI_hi) |>
  mutate(entry = sprintf("%.3f +/- %.3f [%.3f, %.3f]",
                         CDR_mean, CDR_sd, CI_lo, CI_hi)) |>
  select(method, delta, entry) |>
  pivot_wider(names_from=delta, values_from=entry) |>
  as.data.frame() |> print()
sink()

# FIGURES
plot_R_use <- min(plot_R, min(sapply(pools, nrow)))
ref_fig    <- lapply(pools,
                     function(p) p[seq_len(plot_R_use), , drop=FALSE])
md_p_fig   <- lapply(ref_fig, md_fit)
ref_ds_fig <- list(MD = lapply(seq_along(ref_fig),
                               function(j) sort(md_depth(ref_fig[[j]], md_p_fig[[j]]))))

nm_fig       <- sprintf("delta=%.1f", plot_delta)
M_ooc_fig    <- M_ooc_list[[nm_fig]]
chat_ooc_fig <- pred_ooc_list[[nm_fig]]

r_ic_fig  <- compute_r("MD", M_ic, pred_ic,
                       ref_fig, ref_ds_fig, md_p_fig)
r_ooc_fig <- compute_r("MD", M_ooc_fig, chat_ooc_fig,
                       ref_fig, ref_ds_fig, md_p_fig)

all_e   <- rbind(M_ic, M_ooc_fig, ref_fig[[1]], ref_fig[[2]])
grid_df <- expand.grid(
  x = seq(min(all_e[,1])-1, max(all_e[,1])+1, length.out=100),
  y = seq(min(all_e[,2])-1, max(all_e[,2])+1, length.out=100))
grid_mat      <- as.matrix(grid_df)
grid_df$depth <- pmax(md_depth(grid_mat, md_p_fig[[1]]),
                      md_depth(grid_mat, md_p_fig[[2]]))

sc <- bind_rows(
  as_tibble(ref_fig[[1]]) |> rename(e1=V1,e2=V2) |> mutate(g="ref0"),
  as_tibble(ref_fig[[2]]) |> rename(e1=V1,e2=V2) |> mutate(g="ref1"),
  as_tibble(M_ic)         |> rename(e1=V1,e2=V2) |> mutate(g="ic"),
  as_tibble(M_ooc_fig)    |> rename(e1=V1,e2=V2) |> mutate(g="ooc")
) |> mutate(g = factor(g, levels=c("ref0","ref1","ic","ooc")))

base_theme <- theme_classic(base_size=13) + theme(
  plot.title      = element_text(face="bold", size=13),
  legend.position = "bottom",
  panel.border    = element_rect(colour="grey60", fill=NA,
                                 linewidth=0.5))

p_embed <- ggplot() +
  geom_tile(data=grid_df, aes(x, y, fill=depth), alpha=0.55) +
  scale_fill_gradient(low="white", high="#aed6f1",
                      limits=c(0,NA), guide="none") +
  geom_point(data=sc, aes(e1, e2, colour=g, shape=g), alpha=0.85) +
  labs(title="(a) Two-dimensional embedding space",
       x=expression(bold(m)[t*","*1]),
       y=expression(bold(m)[t*","*2])) +
  base_theme

n_ic_fig <- length(r_ic_fig)
cd <- bind_rows(
  tibble(obs=seq_len(n_ic_fig), r=r_ic_fig, grp="ic"),
  tibble(obs=n_ic_fig+seq_along(r_ooc_fig), r=r_ooc_fig, grp="ooc")
) |> mutate(grp=factor(grp, levels=c("ic","ooc")))
sig_pts <- filter(cd, !is.na(r), r <= alpha)

p_chart <- ggplot(cd, aes(obs, r, colour=grp)) +
  annotate("rect", xmin=n_ic_fig+0.5, xmax=nrow(cd)+0.5,
           ymin=-Inf, ymax=Inf, fill="#fadbd8", alpha=0.35) +
  geom_hline(yintercept=alpha, linetype="dashed",
             linewidth=0.7, colour="#922b21") +
  geom_point(size=1.5, alpha=0.70) +
  geom_point(data=sig_pts, aes(obs, r), shape=4, size=3.2,
             stroke=1.1, colour="#922b21", show.legend=FALSE) +
  scale_y_continuous(limits=c(0,1), breaks=seq(0,1,0.2)) +
  labs(title="(b) r-statistic control chart (Mahalanobis depth)",
       x="Observation index (t)",
       y=expression(r(bold(m)[t]))) +
  base_theme

sens_plot_data <- toy_sens_summary |>
  filter(R_size==max(R_sizes), method != "SD") |>
  mutate(method=factor(method, levels=c("MD","HD","PD")))

p_sens <- ggplot(sens_plot_data,
                 aes(x=delta, y=CDR_mean, colour=method,
                     shape=method)) +
  geom_line(linewidth=0.8) +
  geom_point(size=2.5) +
  geom_errorbar(aes(ymin=pmax(CDR_mean-CDR_sd, 0),
                    ymax=pmin(CDR_mean+CDR_sd, 1)),
                width=0.02, linewidth=0.5) +
  geom_hline(yintercept=alpha, linetype="dashed",
             colour="grey40", linewidth=0.6) +
  scale_x_continuous(breaks=delta_levels) +
  scale_y_continuous(limits=c(0,1), breaks=seq(0,1,0.2)) +
  labs(title="Sensitivity: CDR mean \u00b1 SD across 10 OOC resamples",
       subtitle=sprintf("|R_c| = %d,  n_OOC = %d,  10 seeds",
                        max(R_sizes), n_ooc),
       x=expression(paste("Drift level  ", delta)),
       y="CDR (mean \u00b1 SD)",
       colour="Method", shape="Method") +
  theme_classic(base_size=13) +
  theme(legend.position="bottom",
        panel.border=element_rect(colour="grey60", fill=NA,
                                  linewidth=0.5))

ggsave(file.path(out, "toy_fig1_embedding_scatter.png"),
       plot=p_embed,  width=8,  height=7, dpi=300, bg="white")
ggsave(file.path(out, "toy_fig2_control_chart.png"),
       plot=p_chart,  width=9,  height=7, dpi=300, bg="white")
ggsave(file.path(out, "toy_fig3_embedding_and_chart.png"),
       plot=(p_embed | p_chart), width=16, height=7,
       dpi=300, bg="white")
ggsave(file.path(out, "toy_fig4_sensitivity_CDR.png"),
       plot=p_sens,   width=8,  height=6, dpi=300, bg="white")
