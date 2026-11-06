import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


# --- GPU configuration ---
gpu_list = tf.config.experimental.list_physical_devices('GPU')
id_gpu = 1
if gpu_list:
    try:
        tf.config.experimental.set_visible_devices([gpu_list[id_gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpu_list[id_gpu], True)
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")

# =====================================================================
# 0. Global configuration
# =====================================================================
Z_MIN = 0.0
Z_MAX = 4.0
GRID_PATH = "fsps_11band_grid.npz"

TRAIN_CAT = "train_photo.dat"
VALID_CAT = "val_photo.dat"
TEST_CAT  = "test_photo.dat"

SEED = 42

EARLY_STOP_PATIENCE  = 60
EARLY_STOP_MIN_DELTA = 1e-5

CKPT_BEST_WEIGHTS   = "photo_best_sed_model.weights.h5"
CKPT_FINAL_WEIGHTS  = "photo_final_sed_model.weights.h5"
HISTORY_NPZ         = "photo_training_history.npz"
CURVE_PNG           = "photo_training_curve.png"
TEST_FIG_PNG        = "photo_pure_physical_sed_z04.png"
VALID_FIG_PNG       = "photo_validation_sed_z04.png"

FLUX_COLS = [
    'GALEX_NUV_FLUX', 'CFHT_ustar_FLUX', 'HSC_g_FLUX', 'HSC_r_FLUX',
    'HSC_i_FLUX', 'HSC_z_FLUX', 'HSC_y_FLUX', 'UVISTA_J_FLUX',
    'UVISTA_Ks_FLUX', 'IRAC_CH1_FLUX', 'IRAC_CH2_FLUX'
]
ERR_COLS = [c.replace('_FLUX', '_FLUXERR') for c in FLUX_COLS]

# Names of the individual loss terms in the output (order matches the layer return)
LOSS_TERM_NAMES = ["total", "sed", "color", "reg", "var"]


# =====================================================================
# 1. Differentiable multi-template physical SED loss layer
# =====================================================================
class MultiTemplatePhysicsLossLayer(layers.Layer):
    def __init__(self, grid_path="fsps_11band_grid.npz", z_min=0.0, z_max=4.0, **kwargs):
        super(MultiTemplatePhysicsLossLayer, self).__init__(**kwargs)
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid file not found: {grid_path}")

        grid_data = np.load(grid_path)
        raw_flux = grid_data['flux'].astype(np.float32)      # [801, 50, 12, 11]
        raw_z    = grid_data['redshift'].astype(np.float32)  # [801,]

        z_mask = (raw_z >= z_min) & (raw_z <= z_max)
        filtered_flux = raw_flux[z_mask, :, :, :]

        self.grid = tf.constant(filtered_flux, dtype=tf.float32)

        self.Nz     = filtered_flux.shape[0]
        self.Nebv   = filtered_flux.shape[1]
        self.Ntpl   = filtered_flux.shape[2]
        self.Nbands = filtered_flux.shape[3]

        self.z_min = tf.constant(z_min, dtype=tf.float32)
        self.z_max = tf.constant(z_max, dtype=tf.float32)

        self.huber_fn = tf.keras.losses.Huber(
            delta=1.5, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

    def call(self, inputs):
        z_pred, p_ebv, c_tpl, f_obs, f_stat_err = inputs

        f_err_total = tf.maximum(f_stat_err, 1e-5)

        z_clamped = tf.clip_by_value(z_pred, self.z_min, self.z_max)
        z_idx_float = (z_clamped - self.z_min) / (self.z_max - self.z_min + 1e-7) \
                      * tf.cast(self.Nz - 1, tf.float32)

        idx_low  = tf.cast(tf.floor(z_idx_float), tf.int32)
        idx_high = tf.minimum(idx_low + 1, self.Nz - 1)

        w_high = z_idx_float - tf.cast(idx_low, tf.float32)
        w_low  = 1.0 - w_high

        grid_low  = tf.gather(self.grid, idx_low[:, 0])
        grid_high = tf.gather(self.grid, idx_high[:, 0])

        w_low_ext  = w_low[:, :, None, None]
        w_high_ext = w_high[:, :, None, None]
        grid_interp = w_low_ext * grid_low + w_high_ext * grid_high

        grid_ebv_avg = tf.einsum('be,bekc->bkc', p_ebv, grid_interp)
        f_syn        = tf.einsum('bk,bkc->bc', c_tpl, grid_ebv_avg)

        f_norm = tf.reduce_mean(tf.abs(f_obs), axis=-1, keepdims=True)
        dev    = (f_obs - f_syn) / (f_err_total + 0.02 * f_norm)

        l_sed = self.huber_fn(tf.zeros_like(dev), dev)

        # ---------- Adjacent-band color loss ----------
        idx_a = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], tf.int32)
        idx_b = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tf.int32)

        eps = 1e-6
        f_obs_pos = tf.maximum(f_obs, eps)
        f_syn_pos = tf.maximum(f_syn, eps)

        c_obs = (tf.math.log(tf.gather(f_obs_pos, idx_b, axis=-1))
                 - tf.math.log(tf.gather(f_obs_pos, idx_a, axis=-1)))
        c_syn = (tf.math.log(tf.gather(f_syn_pos, idx_b, axis=-1))
                 - tf.math.log(tf.gather(f_syn_pos, idx_a, axis=-1)))

        fa = tf.gather(f_obs, idx_a, axis=-1)
        fb = tf.gather(f_obs, idx_b, axis=-1)
        ea = tf.gather(f_err_total, idx_a, axis=-1)
        eb = tf.gather(f_err_total, idx_b, axis=-1)
        color_err = tf.sqrt((ea / (tf.abs(fa) + eps))**2 +
                            (eb / (tf.abs(fb) + eps))**2)
        color_err = tf.maximum(color_err, 0.05)

        dev_color = (c_obs - c_syn) / color_err

        valid = tf.cast((fa > 0) & (fb > 0), tf.float32)

        d_abs = tf.abs(dev_color)
        huber_elem = tf.where(d_abs <= 1.5,
                              0.5 * d_abs**2,
                              1.5 * (d_abs - 0.75))
        l_color = tf.reduce_sum(huber_elem * valid) / (tf.reduce_sum(valid) + 1e-6)

        # ---------- Template regularization ----------
        l_reg_tpl = 0.005 * tf.reduce_mean(tf.reduce_sum(c_tpl, axis=-1))

        # ---------- E(B-V) ----------
        ebv_grid = tf.linspace(0.0, 1.0, 50)
        ebv_mean = tf.reduce_sum(p_ebv * ebv_grid[None, :], axis=-1, keepdims=True)

        ebv_var = tf.reduce_sum(p_ebv * (ebv_grid[None, :] - ebv_mean)**2, axis=-1)

        l_ebv_var_i = 1.0* tf.reduce_mean(
            tf.square(tf.nn.relu(tf.sqrt(ebv_var + eps) - 0.15)))

        l_ebv_var = 100.0 * l_ebv_var_i

        # ---------- Weighted combination ----------
        # Weights for each term
        l_color_w     = 1.0   * l_color

        l_total = l_sed + l_color_w + l_reg_tpl + l_ebv_var

        # Return: total and the individual terms (all weighted; their sum equals total)
        return (l_total,
                l_sed,
                l_color_w,
                l_reg_tpl,
                l_ebv_var)


# =====================================================================
# 2. Build the pure physical network
# =====================================================================
def build_pure_physical_sed_model(num_bands=11, grid_path="fsps_11band_grid.npz",
                                  z_min=0.0, z_max=4.0):
    flux_input     = layers.Input(shape=(num_bands,), name="flux_input")
    stat_err_input = layers.Input(shape=(num_bands,), name="stat_err_input")

    x_in = layers.Concatenate()([flux_input, stat_err_input])

    x = layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="swish", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x)
    x = layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x)

    z_pred = layers.Dense(
        1, activation="softplus", name="z_pred",
        bias_initializer=tf.keras.initializers.Constant(0.8)
    )(x)
    p_ebv = layers.Dense(50, activation="softmax", name="p_ebv")(x)
    c_tpl = layers.Dense(12, activation="softplus", name="c_tpl")(x)

    # Returns a tuple: (total, sed, color, reg, var)
    loss_outputs = MultiTemplatePhysicsLossLayer(
        grid_path=grid_path, z_min=z_min, z_max=z_max, name="phys_loss"
    )([z_pred, p_ebv, c_tpl, flux_input, stat_err_input])

    (l_total, l_sed, l_color_w, l_reg_tpl, l_ebv_var) = loss_outputs

    model = models.Model(
        inputs=[flux_input, stat_err_input],
        outputs=[z_pred, p_ebv, c_tpl,
                 l_total, l_sed, l_color_w, l_reg_tpl,
                 l_ebv_var]
    )
    return model


# =====================================================================
# 3. Data preprocessing and metric computation
# =====================================================================
def load_and_preprocess_catalog(cat_path, min_redshift=0.0, max_redshift=4.0):
    df = pd.read_csv(cat_path, sep=r'\s+')
    z_col = 'z_ref' if 'z_ref' in df.columns else 'specz'

    valid_mask = (df[z_col] >= min_redshift) & (df[z_col] <= max_redshift) & df[z_col].notna()
    df = df[valid_mask].copy().reset_index(drop=True)

    fluxes = np.nan_to_num(df[FLUX_COLS].values.astype(np.float32),
                           nan=1e-5, posinf=1e4, neginf=1e-5)
    stat_errs = np.nan_to_num(df[ERR_COLS].values.astype(np.float32),
                              nan=1e-5, posinf=1e4, neginf=1e-5)
    z_target = df[z_col].values.reshape(-1, 1).astype(np.float32)

    f_norm = np.mean(np.abs(fluxes), axis=-1, keepdims=True) + 1e-5
    fluxes    = fluxes    / f_norm
    stat_errs = stat_errs / f_norm

    return df, fluxes, stat_errs, z_target


def compute_metrics(z_true, z_pred):
    z_true = z_true.flatten()
    z_pred = z_pred.flatten()
    dz = (z_pred - z_true) / (1.0 + z_true)
    median_dz = np.median(dz)
    sigma_nmad = 1.4826 * np.median(np.abs(dz - median_dz))
    outlier_fraction = np.mean(np.abs(dz) > 0.15) * 100.0
    return sigma_nmad, outlier_fraction, median_dz


def plot_z_scatter(z_true, z_pred, save_path, z_min=0.0, z_max=4.0,
                   title="Photometric Test Sample"):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    z_true = z_true.flatten(); z_pred = z_pred.flatten()
    sigma_nmad, eta, bias = compute_metrics(z_true, z_pred)

    ax.scatter(z_true, z_pred, s=8, alpha=0.3, color='darkgreen',
               edgecolors='none', label='Galaxies')
    line_grid = np.linspace(z_min, z_max, 500)
    ax.plot(line_grid, line_grid, 'r--', lw=1.5, label=r'$z_{\mathrm{phot}} = z_{\mathrm{ref}}$')
    ax.plot(line_grid, line_grid + 0.15 * (1 + line_grid), 'r:', lw=1.2, label=r'$\pm 0.15(1+z)$')
    ax.plot(line_grid, line_grid - 0.15 * (1 + line_grid), 'r:', lw=1.2)

    ax.set_xlim(z_min, z_max); ax.set_ylim(z_min, z_max)
    ax.set_xlabel(r'$z_{\mathrm{ref}}$', fontsize=13)
    ax.set_ylabel(r'$z_{\mathrm{pred}}$', fontsize=13)
    ax.set_title(f"Pure Physical SED Fitting  {title}", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    text_box = (f"$\\sigma_{{\\mathrm{{NMAD}}}} = {sigma_nmad:.4f}$\n"
                f"$\\eta (|\\Delta z|>0.15) = {eta:.2f}\\%$\n"
                f"$\\langle\\Delta z\\rangle = {bias:.4f}$")
    ax.text(0.05, 0.78, text_box, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.85, edgecolor='gray'))

    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout(); plt.savefig(save_path); plt.show()


def plot_training_curve(history, best_epoch=None, save_path=CURVE_PNG):
    train_loss = np.array(history["train_loss"], dtype=np.float64)
    val_loss   = np.array(history["val_loss"], dtype=np.float64)
    ep = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(ep, train_loss, label="Train phys-SED loss", lw=1.3)
    ax.plot(ep, val_loss,   label="Val   phys-SED loss", lw=1.3)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", ls="--", lw=1.2,
                   label=f"Best epoch = {best_epoch}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (log)", fontsize=12)
    ax.set_title("Training / Validation Curve", fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


# =====================================================================
# 4. Training pipeline
# =====================================================================
def train_and_evaluate():
    print("\n==================================================")
    print(" Run mode: [Pure physical SED (unsupervised) + pre-split validation set + early stopping]")
    print(f" Train set (spectroscopic): {TRAIN_CAT}")
    print(f" Validation set (spectroscopic): {VALID_CAT}")
    print(f" Test set (photometric): {TEST_CAT}")
    print("==================================================\n")

    # ---- (A) Load the three datasets ----
    df_train, f_train, e_train, z_train = load_and_preprocess_catalog(
        TRAIN_CAT, Z_MIN, Z_MAX)
    df_val, f_val, e_val, z_val = load_and_preprocess_catalog(
        VALID_CAT, Z_MIN, Z_MAX)
    df_test, f_test, e_test, z_test = load_and_preprocess_catalog(
        TEST_CAT, Z_MIN, Z_MAX)

    print(f"[Train set] N samples = {len(df_train)}")
    print(f"[Validation set] N samples = {len(df_val)}")
    print(f"[Test set] N samples = {len(df_test)}\n")

    # ---- (B) DataLoader ----
    batch_size = 512
    ds_train = (tf.data.Dataset.from_tensor_slices((f_train, e_train))
                .shuffle(min(10000, len(f_train)), seed=SEED)
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    ds_val   = (tf.data.Dataset.from_tensor_slices((f_val, e_val))
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    # ---- (C) Model + optimizer ----
    model = build_pure_physical_sed_model(
        num_bands=11, grid_path=GRID_PATH, z_min=Z_MIN, z_max=Z_MAX)

    epochs = 2000
    steps_per_epoch = len(ds_train)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Note: the model now has 10 outputs
    # 0:z_pred  1:p_ebv  2:c_tpl  3:l_total  4:l_sed  5:l_color_w
    # 6:l_reg   7:l_var

    @tf.function
    def train_step(f, e):
        with tf.GradientTape() as tape:
            outs = model([f, e], training=True)
            l_total = outs[3]
            total_loss = tf.reduce_mean(l_total)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Return total + the individual terms (terms already weighted)
        return (total_loss,
                outs[4], outs[5], outs[6], outs[7])

    @tf.function
    def val_step(f, e):
        outs = model([f, e], training=False)
        l_total = outs[3]
        return (tf.reduce_mean(l_total),
                outs[4], outs[5], outs[6], outs[7])

    # ---- (D) Training loop + early stopping ----
    best_val   = np.inf
    best_epoch = 0
    wait       = 0
    history    = {"train_loss": [], "val_loss": [], "train_terms": [], "val_terms": []}
    nmad_records = []   # [(epoch, sigma, eta, bias), ...]

    print(f"Starting training (EarlyStopping patience={EARLY_STOP_PATIENCE}, "
          f"monitor=val phys-SED loss)...\n")

    for epoch in range(epochs):
        # ---------- train ----------
        train_sums = [0.0] * len(LOSS_TERM_NAMES)   # total + individual terms
        steps = 0
        for f_b, e_b in ds_train:
            outs = train_step(f_b, e_b)
            for i in range(len(LOSS_TERM_NAMES)):
                train_sums[i] += float(outs[i])
            steps += 1
        train_means = [x / max(steps, 1) for x in train_sums]

        # ---------- validation (weighted by batch) ----------
        val_sums = [0.0] * len(LOSS_TERM_NAMES)
        v_count = 0
        for f_b, e_b in ds_val:
            bs = int(tf.shape(f_b)[0])
            outs = val_step(f_b, e_b)
            for i in range(len(LOSS_TERM_NAMES)):
                val_sums[i] += float(outs[i]) * bs
            v_count += bs
        val_means = [x / max(v_count, 1) for x in val_sums]

        t_loss = train_means[0]
        v_loss = val_means[0]

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_terms"].append(list(train_means[1:]))
        history["val_terms"].append(list(val_means[1:]))

        # ---------- early stopping ----------
        improved = v_loss < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val   = v_loss
            best_epoch = epoch + 1
            wait       = 0
            model.save_weights(CKPT_BEST_WEIGHTS)
        else:
            wait += 1

        # ---------- logging ----------
        if epoch == 0 or (epoch + 1) % 10 == 0:
            star = " *" if improved else ""
            print(f"Epoch {epoch+1:03d}/{epochs} | Train: {t_loss:.6f} | "
                  f"Val: {v_loss:.6f} | Best: {best_val:.6f}@{best_epoch}{star}")
            # Individual loss terms (after weighting; their sum equals total)
            def _fmt(name, means):
                return "  ".join(f"{n}={m:.5f}"
                                 for n, m in zip(LOSS_TERM_NAMES[1:], means[1:]))
            print(f"   [Train terms] {_fmt('Train', train_means)}")
            print(f"   [Val   terms] {_fmt('Val',   val_means)}")

        # ---------- print validation NMAD every 20 epochs ----------
        if epoch == 0 or (epoch + 1) % 20 == 0:
            z_vp = model.predict([f_val, e_val], batch_size=512, verbose=0)[0]
            s, eta, b = compute_metrics(z_val, z_vp)
            nmad_records.append([epoch + 1, s, eta, b])
            print(f"   [Val NMAD] σ={s:.4f}  η={eta:.2f}%  bias={b:+.4f}")

        if wait >= EARLY_STOP_PATIENCE:
            print(f"\n>> EarlyStopping triggered: no improvement for {wait} consecutive epochs. "
                  f"Best epoch = {best_epoch}, best val_loss = {best_val:.6f}")
            break

    # ---- (E) Restore the best weights ----
    if os.path.exists(CKPT_BEST_WEIGHTS):
        print(f"\nRestoring best weights: {CKPT_BEST_WEIGHTS}  (epoch {best_epoch})")
        model.load_weights(CKPT_BEST_WEIGHTS)

    # ---- (F) Save training history & curve ----
    np.savez(
        HISTORY_NPZ,
        train_loss=np.array(history["train_loss"], dtype=np.float32),
        val_loss=np.array(history["val_loss"], dtype=np.float32),
        train_terms=np.array(history["train_terms"], dtype=np.float32),
        val_terms=np.array(history["val_terms"], dtype=np.float32),
        loss_term_names=np.array(LOSS_TERM_NAMES[1:]),
        nmad_records=np.array(nmad_records, dtype=np.float32),
        best_epoch=np.int32(best_epoch),
        best_val_loss=np.float32(best_val),
    )
    plot_training_curve(history, best_epoch=best_epoch, save_path=CURVE_PNG)
    print(f"Training history saved: {HISTORY_NPZ}, curve plot: {CURVE_PNG}")

    # ---- (G) Self-check on training and validation sets ----
    print("\n------------------ Evaluation results ------------------")
    z_pred_train = model.predict([f_train, e_train],
                                 batch_size=512, verbose=0)[0]
    s_tr, eta_tr, b_tr = compute_metrics(z_train, z_pred_train)
    print("\n[Train set / spectroscopic sample self-check performance]:")
    print(f"  * σ_NMAD               : {s_tr:.4f}")
    print(f"  * Outlier Fraction     : {eta_tr:.2f}%")
    print(f"  * Median Bias (<Δz>)   : {b_tr:.4f}")

    z_pred_val = model.predict([f_val, e_val],
                               batch_size=512, verbose=0)[0]
    s_va, eta_va, b_va = compute_metrics(z_val, z_pred_val)
    print("\n[Validation set / spectroscopic sample performance]:")
    print(f"  * σ_NMAD               : {s_va:.4f}")
    print(f"  * Outlier Fraction     : {eta_va:.2f}%")
    print(f"  * Median Bias (<Δz>)   : {b_va:.4f}")
    plot_z_scatter(z_val, z_pred_val, save_path=VALID_FIG_PNG,
                   z_min=Z_MIN, z_max=Z_MAX,
                   title="Spectroscopic Validation Sample")

    # ---- (H) Test-set evaluation ----
    z_pred_test = model.predict([f_test, e_test],
                                batch_size=512, verbose=0)[0]
    s_te, eta_te, b_te = compute_metrics(z_test, z_pred_test)
    print("\n[Test set / photometric sample performance ★]:")
    print(f"  * σ_NMAD               : {s_te:.4f}")
    print(f"  * Outlier Fraction     : {eta_te:.2f}%")
    print(f"  * Median Bias (<Δz>)   : {b_te:.4f}")
    plot_z_scatter(z_test, z_pred_test, save_path=TEST_FIG_PNG,
                   z_min=Z_MIN, z_max=Z_MAX,
                   title="Photometric Test Sample")

    # ---- (I) Save final weights ----
    model.save_weights(CKPT_FINAL_WEIGHTS)
    print("\nOutput files:")
    print(f"  Best weights          : {CKPT_BEST_WEIGHTS}")
    print(f"  Final weights         : {CKPT_FINAL_WEIGHTS}")
    print(f"  Training history      : {HISTORY_NPZ}")
    print(f"  Training curve        : {CURVE_PNG}")
    print(f"  Validation scatter    : {VALID_FIG_PNG}")
    print(f"  Test scatter          : {TEST_FIG_PNG}")


if __name__ == "__main__":
    train_and_evaluate()
