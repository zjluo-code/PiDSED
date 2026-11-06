import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================================
# 0. Global configuration and constant definitions
# =====================================================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

CKPT_BEST_WEIGHTS = "supervised_best_weights.weights.h5"
TEST_CAT = "test_photo.dat"       # Modify according to the actual test-set path
OUTPUT_PRED_DAT = "mlp_photo_test_pred.dat" # Output prediction filename

Z_MIN = 0.0
Z_MAX = 4.0

# Survey band column names (must be consistent with the training script)
FLUX_COLS = [
    'GALEX_NUV_FLUX', 'CFHT_ustar_FLUX', 'HSC_g_FLUX', 'HSC_r_FLUX',
    'HSC_i_FLUX', 'HSC_z_FLUX', 'HSC_y_FLUX', 'UVISTA_J_FLUX',
    'UVISTA_Ks_FLUX', 'IRAC_CH1_FLUX', 'IRAC_CH2_FLUX'
]
ERR_COLS = [c.replace('_FLUX', '_FLUXERR') for c in FLUX_COLS]


# =====================================================================
# 1. Build the pure supervised MLP model structure
# =====================================================================
def build_supervised_mlp_model(num_bands=11, z_min=0.0, z_max=4.0):
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

    model = models.Model(
        inputs=[flux_input, stat_err_input],
        outputs=[z_pred]
    )
    return model


# =====================================================================
# 2. Data preprocessing and metric computation
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

    # Per-source normalization (exactly the same as during training)
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
# 3. Inference execution pipeline
# =====================================================================
def run_inference():
    print("\n==================================================")
    print(" Run mode: [Load the best weights and run inference on the test set]")
    print(f" Test-set file: {TEST_CAT}")
    print(f" Weights file: {CKPT_BEST_WEIGHTS}")
    print("==================================================\n")

    if not os.path.exists(CKPT_BEST_WEIGHTS):
        raise FileNotFoundError(f"Weights file '{CKPT_BEST_WEIGHTS}' not found. Please run the training script first to generate model weights!")

    # ---- (A) Load and preprocess the test set ----
    df_test, f_test, e_test, z_test = load_and_preprocess_catalog(
        TEST_CAT, Z_MIN, Z_MAX)
    print(f"[Test set] Number of valid inference samples = {len(df_test)}")

    # ---- (B) Build the model and load the best weights ----
    model = build_supervised_mlp_model(num_bands=len(FLUX_COLS), z_min=Z_MIN, z_max=Z_MAX)
    model.load_weights(CKPT_BEST_WEIGHTS)
    print(f"Weights loaded successfully: {CKPT_BEST_WEIGHTS}")

    # ---- (C) Run batch inference ----
    print("Running forward inference prediction...")
    z_pred_test = model.predict([f_test, e_test], batch_size=512, verbose=1)

    # ---- (D) Compute evaluation metrics ----
    s_te, eta_te, b_te = compute_metrics(z_test, z_pred_test)
    print("\n------------------ Test-set inference metrics ------------------")
    print(f"  * σ_NMAD             : {s_te:.4f}")
    print(f"  * Outlier Fraction   : {eta_te:.2f}%")
    print(f"  * Median Bias (<Δz>) : {b_te:.4f}")
    print("--------------------------------------------------")

    # ---- (E) Save prediction results to test_pred.dat ----
    # Add the predicted redshift values to the DataFrame
    df_test['z_pred'] = z_pred_test.flatten()
    
    # Save as a space-separated .dat file (keep 6 decimal places)
    df_test.to_csv(OUTPUT_PRED_DAT, sep=' ', index=False, float_format='%.6f')
    print(f"\nInference results successfully saved to: {OUTPUT_PRED_DAT}")


if __name__ == "__main__":
    run_inference()
