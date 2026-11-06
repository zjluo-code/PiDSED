import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- GPU configuration ---
gpu_list = tf.config.experimental.list_physical_devices('GPU')
id_gpu = 0
if gpu_list:
    try:
        tf.config.experimental.set_visible_devices([gpu_list[id_gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpu_list[id_gpu], True)
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")

# =====================================================================
# 0. Global configuration and constant definitions
# =====================================================================
BASE_SEED = 42
NUM_RUNS = 20         # Number of independent random training runs

EARLY_STOP_PATIENCE = 100
EARLY_STOP_MIN_DELTA = 1e-4

# Dataset paths and redshift range
#TRAIN_CAT = "../train_photo.dat"
#VALID_CAT = "../val_photo.dat"
#TEST_CAT = "../test_photo.dat"

TRAIN_CAT = "../train_photo_zpred_floor2.dat"
VALID_CAT = "../val_photo_zpred_floor2.dat"
TEST_CAT = "../test_photo.dat"
Z_MIN = 0.0
Z_MAX = 4.0

# Survey band column names
FLUX_COLS = [
    'GALEX_NUV_FLUX', 'CFHT_ustar_FLUX', 'HSC_g_FLUX', 'HSC_r_FLUX',
    'HSC_i_FLUX', 'HSC_z_FLUX', 'HSC_y_FLUX', 'UVISTA_J_FLUX',
    'UVISTA_Ks_FLUX', 'IRAC_CH1_FLUX', 'IRAC_CH2_FLUX'
]
ERR_COLS = [c.replace('_FLUX', '_FLUXERR') for c in FLUX_COLS]


# =====================================================================
# 1. Build the pure supervised MLP model
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
# 3. Single-run training and evaluation function
# =====================================================================
def train_and_evaluate_single_run(run_id, seed, f_train, e_train, z_train,
                                    f_val, e_val, z_val, f_test, e_test, z_test):
    print(f"\n--------------------------------------------------")
    print(f" Starting supervised training run {run_id}/{NUM_RUNS} (Seed = {seed})")
    print(f"--------------------------------------------------")

    # Set random seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ckpt_best = f"supervised_photo_train_photo_test_run_{run_id}_best_weights.weights.h5"

    # DataLoader
    batch_size = 512
    ds_train = (tf.data.Dataset.from_tensor_slices((f_train, e_train, z_train))
                .shuffle(min(10000, len(f_train)), seed=seed)
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    ds_val   = (tf.data.Dataset.from_tensor_slices((f_val, e_val, z_val))
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    # Model + optimizer + loss function
    model = build_supervised_mlp_model(num_bands=11, z_min=Z_MIN, z_max=Z_MAX)

    epochs = 2000
    steps_per_epoch = len(ds_train)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.Huber(delta=1.5)

    @tf.function
    def train_step(f, e, z_true):
        with tf.GradientTape() as tape:
            z_pred = model([f, e], training=True)
            loss_val = loss_fn(z_true, z_pred)
        grads = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_val

    @tf.function
    def val_step(f, e, z_true):
        z_pred = model([f, e], training=False)
        loss_val = loss_fn(z_true, z_pred)
        return loss_val

    def evaluate_dataset_loss(dataset):
        total, count = 0.0, 0
        for f_b, e_b, z_b in dataset:
            bs = int(tf.shape(f_b)[0])
            total += float(val_step(f_b, e_b, z_b)) * bs
            count += bs
        return total / max(count, 1)

    # Training loop + early stopping
    best_val = np.inf
    best_epoch = 0
    wait = 0

    for epoch in range(epochs):
        t_loss, steps = 0.0, 0
        for f_b, e_b, z_b in ds_train:
            t_loss += float(train_step(f_b, e_b, z_b))
            steps += 1
        t_loss /= max(steps, 1)

        v_loss = evaluate_dataset_loss(ds_val)

        improved = v_loss < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val = v_loss
            best_epoch = epoch + 1
            wait = 0
            model.save_weights(ckpt_best)
        else:
            wait += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f} | Best: {best_val:.6f}@{best_epoch}")

        if wait >= EARLY_STOP_PATIENCE:
            print(f">> EarlyStopping triggered at epoch {epoch+1} (best epoch = {best_epoch}, best val_loss = {best_val:.6f})")
            break

    # Restore the best weights
    if os.path.exists(ckpt_best):
        model.load_weights(ckpt_best)

    # Test-set evaluation
    z_pred_test = model.predict([f_test, e_test], batch_size=512, verbose=0)
    s_te, eta_te, b_te = compute_metrics(z_test, z_pred_test)

    print(f"[Run {run_id} test-set results] σ_NMAD = {s_te:.4f}, η = {eta_te:.2f}%, bias = {b_te:+.4f}")
    print(f"[Info] Best model for run {run_id} has been saved: {ckpt_best}")

    # Clear GPU memory and backend graph state
    tf.keras.backend.clear_session()

    return s_te, eta_te, b_te


# =====================================================================
# 4. Main control function: 20 independent random training runs and statistical summary
# =====================================================================
def main():
    print("\n==================================================")
    print(f" Run mode: [Pure supervised MLP model, 20 independent random training runs and statistics]")
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
            f_train=f_train, e_train=e_train, z_train=z_train,
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
    print("All best-model weight files from the 20 independent runs (supervised_run_1_best_weights.weights.h5 ~ supervised_run_20_best_weights.weights.h5) have been saved in the current directory!\n")


if __name__ == "__main__":
    main()
