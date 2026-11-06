import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
TEST_CAT = "test_photo.dat"         # Modify according to the actual test-set path
#CKPT_BEST_WEIGHTS = "best_sed_model.weights.h5"
CKPT_BEST_WEIGHTS = "./random_run/piDSED_photo_train_photo_test_run_2_best_weights.h5"
OUTPUT_PRED_DAT = "phy_photo_test_pred_cal_time.dat" # Output prediction filename

FLUX_COLS = [
    'GALEX_NUV_FLUX', 'CFHT_ustar_FLUX', 'HSC_g_FLUX', 'HSC_r_FLUX',
    'HSC_i_FLUX', 'HSC_z_FLUX', 'HSC_y_FLUX', 'UVISTA_J_FLUX',
    'UVISTA_Ks_FLUX', 'IRAC_CH1_FLUX', 'IRAC_CH2_FLUX'
]
ERR_COLS = [c.replace('_FLUX', '_FLUXERR') for c in FLUX_COLS]


# =====================================================================
# 1. Differentiable multi-template physical SED loss layer (required for model reconstruction)
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
# 2. Build the pure physical network structure
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
# 4. Inference execution pipeline
# =====================================================================
def run_inference():
    print("\n==================================================")
    print(" Run mode: [Physical SED model inference on the test set]")
    print(f" Test-set file: {TEST_CAT}")
    print(f" Weights file: {CKPT_BEST_WEIGHTS}")
    print(f" Output file: {OUTPUT_PRED_DAT}")
    print("==================================================\n")

    if not os.path.exists(CKPT_BEST_WEIGHTS):
        raise FileNotFoundError(f"Weights file '{CKPT_BEST_WEIGHTS}' not found. Please run the physical model training script first!")

    # ---- (A) Load and preprocess the test set ----
    df_test, f_test, e_test, z_test = load_and_preprocess_catalog(
        TEST_CAT, Z_MIN, Z_MAX)
    print(f"[Test set] Number of valid inference samples = {len(df_test)}")

    # ---- (B) Build the model and load the best weights ----
    model = build_pure_physical_sed_model(
        num_bands=len(FLUX_COLS), grid_path=GRID_PATH, z_min=Z_MIN, z_max=Z_MAX)
    model.load_weights(CKPT_BEST_WEIGHTS)
    print(f"Best weights loaded successfully: {CKPT_BEST_WEIGHTS}")

    # ---- (C) Forward inference to predict redshifts ----
    print("Running batch inference with the physical network...")
    preds = model.predict([f_test, e_test], batch_size=512, verbose=1)
    z_pred_test = preds[0]  # Output list item 0 is z_pred

    # ---- (D) Compute evaluation metrics ----
    s_te, eta_te, b_te = compute_metrics(z_test, z_pred_test)
    print("\n------------------ Test-set inference metrics ------------------")
    print(f"  * σ_NMAD             : {s_te:.4f}")
    print(f"  * Outlier Fraction   : {eta_te:.2f}%")
    print(f"  * Median Bias (<Δz>) : {b_te:.4f}")
    print("--------------------------------------------------")

    # ---- (E) Save prediction results to phy_test_pred.dat ----
    df_test['z_pred'] = z_pred_test.flatten()
    
    # Save as a space-separated .dat file (keep 6 decimal places)
    df_test.to_csv(OUTPUT_PRED_DAT, sep=' ', index=False, float_format='%.6f')
    print(f"\nInference results successfully saved to: {OUTPUT_PRED_DAT}")


if __name__ == "__main__":
    run_inference()
