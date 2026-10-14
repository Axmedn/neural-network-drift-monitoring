suppressPackageStartupMessages({
  library(keras3); library(tensorflow); library(ddalpha)
  library(dplyr); library(tibble); library(tidyr)
  library(ggplot2)
})

try(tensorflow::tf$config$experimental$enable_op_determinism(), silent = TRUE)
SEED <- 1
set.seed(SEED); tensorflow::set_random_seed(SEED)

#CONFIG
CACHE_VERSION <- "v2"
FAST_DEV      <- FALSE
alpha         <- 0.05
sigma_levels  <- c(0.1, 0.2, 0.3, 0.4)
OOC_SEEDS     <- 1:10
N_BOOT        <- 1000

if (FAST_DEV) {
  R_sizes <- c(200, 400, 800); n_ooc <- 200
  epochs  <- 5; batch_size <- 128; patience <- 3
} else {
  R_sizes <- c(2000, 3000, 4000); n_ooc <- 500
  epochs  <- 20; batch_size <- 128; patience <- 3
}

embed_dim <- 16
HDr_B     <- 500
PD3_B     <- 1000
B_PDa3    <- 500
methods   <- c("MD", "HDr", "PD3", "PDa3")

# ── OUTPUT FOLDER
out <- file.path(getwd(), "mnist_experiment_outputs")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

# ── CACHE PATHS
cache_file <- file.path(getwd(), sprintf(
  "mnist_cache_%s_FAST=%s_ep=%d_emb=%d_nooc=%d_seed=%d.rds",
  CACHE_VERSION, FAST_DEV, epochs, embed_dim, n_ooc, SEED))
model_file       <- paste0(cache_file, "_model.keras")
embed_model_file <- paste0(cache_file, "_embed.keras")

# HELPERS
conf_from_prob <- function(prob)
  if (is.null(dim(prob))) pmax(prob, 1-prob) else apply(prob, 1, max)

weighted_rate <- function(sig, chat) {
  n <- length(sig)
  sum(vapply(sort(unique(chat)), function(cl) {
    idx <- chat == cl & !is.na(sig)
    if (!any(idx)) 0 else (sum(chat==cl)/n) * mean(sig[idx])
  }, numeric(1)))
}

r_stat    <- function(d_new, d_sorted)
  findInterval(d_new, d_sorted, rightmost.closed=TRUE) / length(d_sorted)

far_class <- function(d_ref)
  mean(r_stat(d_ref, sort(d_ref)) <= alpha)

boot_ci_lo <- function(x, B=N_BOOT, conf=0.95) {
  if (all(is.na(x))) return(NA_real_)
  x  <- x[!is.na(x)]
  bm <- replicate(B, mean(sample(x, length(x), replace=TRUE)))
  round(quantile(bm, (1-conf)/2, names=FALSE), 3)
}
boot_ci_hi <- function(x, B=N_BOOT, conf=0.95) {
  if (all(is.na(x))) return(NA_real_)
  x  <- x[!is.na(x)]
  bm <- replicate(B, mean(sample(x, length(x), replace=TRUE)))
  round(quantile(bm, 1-(1-conf)/2, names=FALSE), 3)
}

# ── DATA + TRAIN + EMBED (CACHED) ────────────────────────────
if (file.exists(cache_file) &&
    file.exists(model_file) &&
    file.exists(embed_model_file)) {

  cache        <- readRDS(cache_file)
  x_train_raw  <- cache$x_train_raw
  x_test_raw   <- cache$x_test_raw
  y_train      <- cache$y_train
  pred_train   <- cache$pred_train
  conf_train   <- cache$conf_train
  ok_idx       <- cache$ok_idx
  y_ok         <- cache$y_ok
  conf_ok      <- cache$conf_ok
  M_ok         <- cache$M_ok
  y_test       <- cache$y_test
  pred_test    <- cache$pred_test
  ic_correct   <- cache$ic_correct
  M_test       <- cache$M_test
  acc_tr       <- cache$acc_tr
  acc_te       <- cache$acc_te
  model        <- load_model(model_file)
  embed_model  <- load_model(embed_model_file)
  
} else {

  mnist        <- dataset_mnist()
  x_train_raw  <- array_reshape(mnist$train$x, c(60000, 784)) / 255
  x_test_raw   <- array_reshape(mnist$test$x,  c(10000, 784)) / 255
  y_train      <- as.integer(mnist$train$y)
  y_test       <- as.integer(mnist$test$y)
  
  if (FAST_DEV) {
    set.seed(SEED)
    tr_idx      <- sample(60000, 15000)
    te_idx      <- sample(10000,  3000)
    x_train_raw <- x_train_raw[tr_idx,]; y_train <- y_train[tr_idx]
    x_test_raw  <- x_test_raw[te_idx,];  y_test  <- y_test[te_idx]
  }
  
  model <- keras_model_sequential() |>
    layer_dense(256, activation="relu",   input_shape=784, name="h1") |>
    layer_dense(64,  activation="relu",                    name="h2") |>
    layer_dense(embed_dim, activation="linear",            name="embed") |>
    layer_dense(10,  activation="softmax",                 name="out")
  
  model |> compile(optimizer_adam(0.001),
                   loss="sparse_categorical_crossentropy",
                   metrics="accuracy")
  
  model |> fit(x_train_raw, y_train,
               epochs=epochs, batch_size=batch_size,
               validation_split=0.1,
               callbacks=list(callback_early_stopping(
                 "val_accuracy", patience=patience,
                 restore_best_weights=TRUE)),
               verbose=1)
  
  embed_model <- keras_model(model$inputs,
                             get_layer(model, "embed")$output)
  
  prob_train <- model |> predict(x_train_raw, verbose=0)
  pred_train <- apply(prob_train, 1, which.max) - 1L
  conf_train <- conf_from_prob(prob_train)
  acc_tr     <- mean(pred_train == y_train)
  
  ok_idx  <- which(pred_train == y_train)
  M_ok    <- embed_model |> predict(x_train_raw[ok_idx,], verbose=0)
  y_ok    <- y_train[ok_idx]
  conf_ok <- conf_train[ok_idx]
  
  prob_test  <- model |> predict(x_test_raw, verbose=0)
  pred_test  <- apply(prob_test, 1, which.max) - 1L
  ic_correct <- pred_test == y_test
  acc_te     <- mean(ic_correct)
  M_test     <- embed_model |> predict(x_test_raw, verbose=0)
  
  saveRDS(list(
    x_train_raw=x_train_raw, x_test_raw=x_test_raw,
    y_train=y_train, pred_train=pred_train, conf_train=conf_train,
    ok_idx=ok_idx, y_ok=y_ok, conf_ok=conf_ok, M_ok=M_ok,
    y_test=y_test, pred_test=pred_test,
    ic_correct=ic_correct, M_test=M_test,
    acc_tr=acc_tr, acc_te=acc_te
  ), cache_file)
  save_model(model,       model_file)
  save_model(embed_model, embed_model_file)
  cat("Cache saved.\n")
}

cat(sprintf("\nTrain acc: %.2f%%  |  Test acc: %.2f%%\n",
            100*acc_tr, 100*acc_te))

fashion <- dataset_fashion_mnist()
x_fash  <- array_reshape(fashion$test$x, c(10000, 784)) / 255

# REFERENCE POOLS
use_confidence <- any(pred_train != y_train)

pools <- lapply(0:9, function(cls) {
  idx <- which(y_ok == cls)
  if (use_confidence) idx <- idx[order(conf_ok[idx], decreasing=TRUE)]
  M_ok[idx, , drop=FALSE]
})

stopifnot("Pool too small" = min(sapply(pools, nrow)) >= max(R_sizes))
cat(sprintf("Reference pooling: confidence-ordering = %s\n",
            ifelse(use_confidence, "ON", "OFF")))

# DEPTH FUNCTIONS
md_fit <- function(R, ridge=1e-6)
  list(mu=colMeans(R), Si=solve(cov(R) + diag(ridge, ncol(R))))

md_score <- function(X, p) {
  Z <- sweep(as.matrix(X), 2, p$mu, "-")
  1 / (1 + rowSums((Z %*% p$Si) * Z))
}

hdr_score <- function(Xq, R, B=HDr_B, seed=SEED) {
  Xq <- as.matrix(Xq); R <- as.matrix(R)
  if (nrow(R) <= ncol(R)) return(rep(NA_real_, nrow(Xq)))
  eps <- 1e-4 * median(sqrt(rowSums(R^2))) + 1e-8
  set.seed(seed)
  Rj  <- R  + matrix(rnorm(length(R),  0, eps), nrow=nrow(R))
  Xqj <- Xq + matrix(rnorm(length(Xq), 0, eps), nrow=nrow(Xq))
  out <- tryCatch(
    ddalpha::depth.halfspace(Xqj, Rj, exact=FALSE,
                             num.directions=B, seed=seed),
    error=function(e) rep(NA_real_, nrow(Xq)))
  as.vector(out)
}

pd3_score <- function(Xq, R, B=PD3_B, seed=SEED)
  tryCatch(
    as.vector(ddalpha::depth.projection(
      as.matrix(Xq), as.matrix(R),
      method="random", num.directions=B, seed=seed)),
    error=function(e) rep(NA_real_, nrow(as.matrix(Xq))))

set.seed(999)
tmp       <- matrix(rnorm(B_PDa3 * embed_dim), nrow=B_PDa3)
dirs_pda3 <- tmp / sqrt(rowSums(tmp^2))

pda3_fit <- function(R) {
  R   <- as.matrix(R)
  P   <- tcrossprod(dirs_pda3, R)
  med <- apply(P, 1, median)
  mad <- apply(P, 1, function(v) median(abs(v - median(v))))
  mad[mad < 1e-10] <- NA_real_
  list(med=med, mad=mad)
}

pda3_score <- function(Xq, fit) {
  Xq <- as.matrix(Xq)
  O  <- tcrossprod(Xq, dirs_pda3)
  O  <- sweep(O, 2, fit$med, "-")
  O  <- sweep(pmax(O, 0), 2, fit$mad, "/")
  O[!is.finite(O)] <- 0
  1 / (1 + apply(O, 1, max))
}

compute_r <- function(m, M, chat, ref, ref_ds, md_p, pda3_p,
                      pd_seed=SEED) {
  r <- rep(NA_real_, nrow(M))
  for (cls in 0:9) {
    idx <- which(chat == cls); if (!length(idx)) next
    Xq  <- M[idx, , drop=FALSE]
    dn  <- switch(m,
                  MD   = md_score(Xq, md_p[[cls+1]]),
                  HDr  = hdr_score(Xq, ref[[cls+1]], seed=pd_seed),
                  PD3  = pd3_score(Xq, ref[[cls+1]], seed=pd_seed),
                  PDa3 = pda3_score(Xq, pda3_p[[cls+1]])
    )
    ok <- !is.na(dn)
    if (any(ok))
      r[idx[ok]] <- r_stat(dn[ok], ref_ds[[m]][[cls+1]])
  }
  r
}

build_ref <- function(R) {
  ref    <- lapply(pools, function(p) p[seq_len(R), , drop=FALSE])
  md_p   <- lapply(ref, md_fit)
  pda3_p <- lapply(ref, pda3_fit)
  ref_depths <- list(
    MD   = lapply(seq_along(ref),
                  function(j) md_score(ref[[j]], md_p[[j]])),
    HDr  = lapply(ref, function(Rc) {
      v <- hdr_score(Rc, Rc); if (anyNA(v)) NULL else v }),
    PD3  = lapply(ref, function(Rc) {
      v <- pd3_score(Rc, Rc); if (anyNA(v)) NULL else v }),
    PDa3 = lapply(seq_along(ref),
                  function(j) pda3_score(ref[[j]], pda3_p[[j]]))
  )
  ref_ds <- lapply(ref_depths, function(dl)
    lapply(dl, function(v) if (is.null(v)) NULL else sort(v)))
  FAR <- vapply(methods, function(m) {
    dl <- ref_depths[[m]]
    if (is.null(dl) || any(sapply(dl, is.null))) return(NA_real_)
    mean(vapply(dl, far_class, numeric(1)))
  }, numeric(1))
  list(ref=ref, md_p=md_p, pda3_p=pda3_p, ref_ds=ref_ds, FAR=FAR)
}

#FIXED OOC STREAMS
set.seed(2024); ooc_idx_fixed  <- sample(seq_len(nrow(x_test_raw)), n_ooc)
set.seed(42);   eps_base_fixed <- matrix(rnorm(n_ooc*784), nrow=n_ooc)

ooc_fixed <- lapply(sigma_levels, function(sig) {
  Xn   <- pmax(pmin(
    x_test_raw[ooc_idx_fixed,] + sig*eps_base_fixed, 1), 0)
  prob <- model |> predict(Xn, verbose=0)
  chat <- apply(prob, 1, which.max) - 1L
  list(M=embed_model |> predict(Xn, verbose=0),
       c_hat=chat, sigma=sig)
})
names(ooc_fixed) <- sprintf("Gaussian_s%.1f", sigma_levels)

set.seed(9999); fi_fixed <- sample(seq_len(nrow(x_fash)), n_ooc)
prob_fash_fixed <- model |> predict(x_fash[fi_fixed,], verbose=0)
ooc_fixed[["FashionMNIST"]] <- list(
  M     = embed_model |> predict(x_fash[fi_fixed,], verbose=0),
  c_hat = apply(prob_fash_fixed, 1, which.max) - 1L,
  sigma = NA_real_)

#SINGLE-RUN EVALUATION
mnist_results <- bind_rows(lapply(R_sizes, function(R) {
  cat(sprintf("\n|R_c| = %d\n", R))
  rb    <- build_ref(R)
  cat(sprintf("  FAR: %s\n",
              paste(sprintf("%s=%.3f", methods, rb$FAR), collapse="  ")))
  valid <- names(rb$FAR)[!is.na(rb$FAR) & abs(rb$FAR-alpha) <= 0.02]
  
  ic_rows <- bind_rows(lapply(methods, function(m) {
    if (!(m %in% valid))
      return(tibble(R_size=R, method=m, cond="IC", sigma=NA_real_,
                    FAR=NA_real_, SR=NA_real_, SR_M=NA_real_,
                    SR_C=NA_real_, CDR=NA_real_, phase="IC"))
    r   <- compute_r(m, M_test, pred_test,
                     rb$ref, rb$ref_ds, rb$md_p, rb$pda3_p)
    sig <- !is.na(r) & r <= alpha
    tibble(R_size=R, method=m, cond="IC", sigma=NA_real_,
           FAR=rb$FAR[m],
           SR  =weighted_rate(sig, pred_test),
           SR_M=mean(sig[!ic_correct], na.rm=TRUE),
           SR_C=mean(sig[ ic_correct], na.rm=TRUE),
           CDR=NA_real_, phase="IC")
  }))
  
  ooc_rows <- bind_rows(lapply(names(ooc_fixed), function(nm) {
    od <- ooc_fixed[[nm]]
    bind_rows(lapply(methods, function(m) {
      if (!(m %in% valid))
        return(tibble(R_size=R, method=m, cond=nm, sigma=od$sigma,
                      FAR=NA_real_, SR=NA_real_, SR_M=NA_real_,
                      SR_C=NA_real_, CDR=NA_real_, phase="OOC"))
      r   <- compute_r(m, od$M, od$c_hat,
                       rb$ref, rb$ref_ds, rb$md_p, rb$pda3_p)
      sig <- !is.na(r) & r <= alpha
      tibble(R_size=R, method=m, cond=nm, sigma=od$sigma,
             FAR=rb$FAR[m], SR=NA_real_, SR_M=NA_real_, SR_C=NA_real_,
             CDR=weighted_rate(sig, od$c_hat), phase="OOC")
    }))
  }))
  
  bind_rows(ic_rows, ooc_rows)
}))

mnist_res <- mnist_results |>
  mutate(across(c(FAR, SR, SR_M, SR_C, CDR), \(x) round(x, 3)))

mnist_res |> filter(phase=="IC") |>
  select(method, R_size, FAR, SR, SR_M, SR_C) |>
  arrange(method, R_size) |> print(n=Inf)

mnist_res |> filter(phase=="OOC", R_size==max(R_sizes)) |>
  select(method, cond, CDR) |>
  pivot_wider(names_from=cond, values_from=CDR) |> print()

# SENSITIVITY ANALYSIS

mnist_sens_results <- bind_rows(lapply(R_sizes, function(R) {
  cat(sprintf("\n  |R_c| = %d\n", R))
  rb_s    <- build_ref(R)
  valid_s <- names(rb_s$FAR)[!is.na(rb_s$FAR) &
                               abs(rb_s$FAR-alpha) <= 0.02]
  
  bind_rows(lapply(OOC_SEEDS, function(s) {
    cat(sprintf("    seed %d\n", s))
    set.seed(s)
    ooc_idx_s  <- sample(seq_len(nrow(x_test_raw)), n_ooc)
    eps_base_s <- matrix(rnorm(n_ooc*784), nrow=n_ooc)
    
    gaussian_s <- lapply(sigma_levels, function(sig) {
      Xn   <- pmax(pmin(
        x_test_raw[ooc_idx_s,] + sig*eps_base_s, 1), 0)
      prob <- model |> predict(Xn, verbose=0)
      chat <- apply(prob, 1, which.max) - 1L
      list(M=embed_model |> predict(Xn, verbose=0),
           c_hat=chat, sigma=sig)
    })
    names(gaussian_s) <- sprintf("Gaussian_s%.1f", sigma_levels)
    
    set.seed(s + 1000)
    fi_s    <- sample(seq_len(nrow(x_fash)), n_ooc)
    prob_fs <- model |> predict(x_fash[fi_s,], verbose=0)
    fashion_s <- list(
      M     = embed_model |> predict(x_fash[fi_s,], verbose=0),
      c_hat = apply(prob_fs, 1, which.max) - 1L,
      sigma = NA_real_)
    
    all_ooc_s <- c(gaussian_s, list(FashionMNIST=fashion_s))
    
    bind_rows(lapply(names(all_ooc_s), function(nm) {
      od <- all_ooc_s[[nm]]
      bind_rows(lapply(valid_s, function(m) {
        r   <- compute_r(m, od$M, od$c_hat,
                         rb_s$ref, rb_s$ref_ds,
                         rb_s$md_p, rb_s$pda3_p,
                         pd_seed=s)
        sig <- !is.na(r) & r <= alpha
        tibble(R_size=R, seed=s, method=m,
               cond=nm, sigma=od$sigma,
               CDR=weighted_rate(sig, od$c_hat))
      }))
    }))
  }))
}))

mnist_sens_summary <- mnist_sens_results |>
  group_by(R_size, method, cond, sigma) |>
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
  mnist_sens_summary |>
    filter(R_size==R, grepl("^Gaussian", cond)) |>
    select(method, sigma, CDR_report) |>
    pivot_wider(names_from=sigma, values_from=CDR_report) |> print()
}

mnist_sens_summary |>
  filter(cond=="FashionMNIST", R_size==max(R_sizes)) |>
  select(method, CDR_mean, CDR_sd, CI_lo, CI_hi) |> print()

# ── SAVE ALL OUTPUTS
mnist_table1 <- mnist_res |> filter(phase=="IC") |>
  select(method, R_size, FAR, SR, SR_M, SR_C) |>
  arrange(method, R_size)

mnist_table2 <- mnist_res |>
  filter(phase=="OOC", R_size==max(R_sizes)) |>
  select(method, cond, CDR) |>
  pivot_wider(names_from=cond, values_from=CDR)

write.csv(mnist_table1,
          file.path(out, "mnist_table1_FAR_SR.csv"),
          row.names=FALSE)
write.csv(mnist_table2,
          file.path(out, "mnist_table2_CDR_singlerun.csv"),
          row.names=FALSE)
write.csv(mnist_sens_summary,
          file.path(out, "mnist_sensitivity_full.csv"),
          row.names=FALSE)
saveRDS(mnist_sens_results,
        file.path(out, "mnist_sensitivity_raw.rds"))
saveRDS(mnist_sens_summary,
        file.path(out, "mnist_sensitivity_summary.rds"))

sink(file.path(out, "mnist_latex_numbers.txt"))

print(as.data.frame(mnist_table1))
cat("\nTABLE 2: Single-run CDR, |R_c| =", max(R_sizes), "\n")
print(as.data.frame(mnist_table2))
mnist_sens_summary |>
  filter(R_size==max(R_sizes)) |>
  mutate(entry = sprintf("%.3f +/- %.3f [%.3f, %.3f]",
                         CDR_mean, CDR_sd, CI_lo, CI_hi)) |>
  select(method, cond, entry) |>
  pivot_wider(names_from=cond, values_from=entry) |>
  as.data.frame() |> print()
sink()
