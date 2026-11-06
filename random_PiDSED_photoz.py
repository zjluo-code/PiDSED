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
GRID_PATH = "../fsps_11band_grid.npz"

TRAIN_CAT = "../train_photo.dat"
VALID_CAT = "../val_photo.dat"
TEST_CAT  = "../test_photo.dat"

BASE_SEED = 42
NUM_RUNS = 20         # Number of independent random training runs

EARLY_STOP_PATIENCE  = 100
EARLY_STOP_MIN_DELTA = 1e-5

FLUX_COLS = [
    'GALEX_NUV_FLUX', 'CFHT_ustar_FLUX', 'HSC_g_FLUX', 'HSC_r_FLUX',
    'HSC_i_FLUX', 'HSC_z_FLUX', 'HSC_y_FLUX', 'UVISTA_J_FLUX',
    'UVISTA_Ks_FLUX', 'IRAC_CH1_FLUX', 'IRAC_CH2_FLUX'
]
ERR_COLS = [c.replace('_FLUX', '_FLUXERR') for c in FLUX_COLS]

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
        self.Nz   = filtered_flux.shape[0]
        self.Nebv = filtered_flux.shape[1]
        self.Ntpl = filtered_flux.shape[2]
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

        # ---------- E(B-V) variance constraint ----------
        ebv_grid = tf.linspace(0.0, 1.0, 50)
        ebv_mean = tf.reduce_sum(p_ebv * ebv_grid[None, :], axis=-1, keepdims=True)
        ebv_var = tf.reduce_sum(p_ebv * (ebv_grid[None, :] - ebv_mean)**2, axis=-1)
        l_ebv_var = 100 * tf.reduce_mean(
            tf.square(tf.nn.relu(tf.sqrt(ebv_var + eps) - 0.15)))

        l_color_w = 1.0 * l_color
        l_total = l_sed + l_color_w + l_reg_tpl + l_ebv_var

        return (l_total, l_sed, l_color_w, l_reg_tpl, l_ebv_var)


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


# =====================================================================
# 4. Single-run training and evaluation function
# =====================================================================
def train_and_evaluate_single_run(run_id, seed, f_train, e_train, f_val, e_val, z_val, f_test, e_test, z_test):
    print(f"\n--------------------------------------------------")
    print(f" Starting training run {run_id}/{NUM_RUNS} (Seed = {seed})")
    print(f"--------------------------------------------------")

    # Set random seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Best-weight save path for each run (kept permanently, not deleted)
    ckpt_best = f"piDSED_photo_test_run_{run_id}_best_weights.h5"

    # DataLoader
    batch_size = 512
    ds_train = (tf.data.Dataset.from_tensor_slices((f_train, e_train))
                .shuffle(min(10000, len(f_train)), seed=seed)
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    ds_val   = (tf.data.Dataset.from_tensor_slices((f_val, e_val))
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    # Model + optimizer
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

    @tf.function
    def train_step(f, e):
        with tf.GradientTape() as tape:
            outs = model([f, e], training=True)
            l_total = outs[3]
            total_loss = tf.reduce_mean(l_total)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return (total_loss, outs[4], outs[5], outs[6], outs[7])

    @tf.function
    def val_step(f, e):
        outs = model([f, e], training=False)
        l_total = outs[3]
        return (tf.reduce_mean(l_total), outs[4], outs[5], outs[6], outs[7])

    # Training loop + early stopping
    best_val = np.inf
    best_epoch = 0
    wait = 0

    for epoch in range(epochs):
        # train
        train_sums = [0.0] * len(LOSS_TERM_NAMES)
        steps = 0
        for f_b, e_b in ds_train:
            outs = train_step(f_b, e_b)
            for i in range(len(LOSS_TERM_NAMES)):
                train_sums[i] += float(outs[i])
            steps += 1
        train_means = [x / max(steps, 1) for x in train_sums]

        # validation
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

        improved = v_loss < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val = v_loss
            best_epoch = epoch + 1
            wait = 0
            model.save_weights(ckpt_best)
            print(f"  [Epoch {epoch+1:03d}] ★ Found better validation loss: {best_val:.6f} -> saved to {ckpt_best}")
        else:
            wait += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train: {t_loss:.6f} | Val: {v_loss:.6f} | Best: {best_val:.6f}@{best_epoch}")

        if wait >= EARLY_STOP_PATIENCE:
            print(f">> EarlyStopping triggered at epoch {epoch+1} (best epoch = {best_epoch}, best val_loss = {best_val:.6f})")
            break

    # Restore the best weights
    if os.path.exists(ckpt_best):
        model.load_weights(ckpt_best)

    # Test-set inference evaluation
    z_pred_test = model.predict([f_test, e_test], batch_size=512, verbose=0)[0]
    s_te, eta_te, b_te = compute_metrics(z_test, z_pred_test)

    print(f"[Run {run_id} test-set results] σ_NMAD = {s_te:.4f}, η = {eta_te:.2f}%, bias = {b_te:.4f}")
    print(f"[Info] Best model weights from run {run_id} have been safely retained: {ckpt_best}")

    # Clear backend graph state to prevent GPU memory leaks
    tf.keras.backend.clear_session()

    return s_te, eta_te, b_te


# =====================================================================
# 5. Main control flow: 20 independent random training runs and statistical summary
# =====================================================================
def main():
    print("\n==================================================")
    print(f" Run mode: [PiDSED model, 20 independent random training runs with model retention]")
    print(f" Train set: {TRAIN_CAT}")
    print(f" Validation set: {VALID_CAT}")
    print(f" Test set: {TEST_CAT}")
    print(f" Total number of independent runs: {NUM_RUNS}")
    print("==================================================\n")

    # Load datasets (all runs share the same data split)
    df_train, f_train, e_train, z_train = load_and_preprocess_catalog(TRAIN_CAT, Z_MIN, Z_MAX)
    df_val, f_val, e_val, z_val = load_and_preprocess_catalog(VALID_CAT, Z_MIN, Z_MAX)
    df_test, f_test, e_test, z_test = load_and_preprocess_catalog(TEST_CAT, Z_MIN, Z_MAX)

    nmad_list = []
    eta_list = []
    bias_list = []

    for run_id in range(1, NUM_RUNS + 1):
        seed = BASE_SEED + run_id * 10
        s, eta, b = train_and_evaluate_single_run(
            run_id=run_id,
            seed=seed,
            f_train=f_train, e_train=e_train,
            f_val=f_val, e_val=e_val, z_val=z_val,
            f_test=f_test, e_test=e_test, z_test=z_test
        )
        nmad_list.append(s)
        eta_list.append(eta)
        bias_list.append(b)

    # Compute mean and standard deviation
    nmad_mean, nmad_std = np.mean(nmad_list), np.std(nmad_list, ddof=1)
    eta_mean, eta_std = np.mean(eta_list), np.std(eta_list, ddof=1)
    bias_mean, bias_std = np.mean(bias_list), np.std(bias_list, ddof=1)

    print("\n==================================================")
    print(" ★ Statistical summary over 20 independent random training runs (Mean ± Std) ★")
    print("==================================================")
    print(f" * σ_NMAD             : {nmad_mean:.4f} ± {nmad_std:.4f}")
    print(f" * Outlier Fraction (η) : {eta_mean:.2f}% ± {eta_std:.2f}%")
    print(f" * Median Bias (<Δz>)   : {bias_mean:+.4f} ± {bias_std:.4f}")
    print("==================================================\n")
    print("All best-model weight files from the 20 independent runs (pi_dosed_run_1_best_weights.h5 ~ pi_dosed_run_20_best_weights.h5) have been saved in the current directory!\n")


if __name__ == "__main__":
    main()
