import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =====================================================================
# 0. Global configuration and constant definitions (must be placed before all functions and calls!)
# =====================================================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

EARLY_STOP_PATIENCE = 60
EARLY_STOP_MIN_DELTA = 1e-4

# File paths and weight-saving names
CKPT_BEST_WEIGHTS = "supervised_best_weights.weights.h5"
CKPT_FINAL_WEIGHTS = "supervised_final_weights.weights.h5"
HISTORY_NPZ = "supervised_history.npz"
CURVE_PNG = "supervised_training_curve.png"
VALID_FIG_PNG = "supervised_val_scatter.png"
TEST_FIG_PNG = "supervised_test_scatter.png"

# Dataset paths and redshift range (modify according to actual filenames)
TRAIN_CAT = "train_spec.dat"
VALID_CAT = "val_spec.dat"
TEST_CAT = "test_spec.dat"
Z_MIN = 0.0
Z_MAX = 4.0

# Survey band column names (11 bands and corresponding error columns; modify according to actual data header)
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

    # Concatenate fluxes and statistical errors as input features
    x_in = layers.Concatenate()([flux_input, stat_err_input])

    x = layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="swish", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x)
    x = layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x)

    # The pure supervised model directly outputs the redshift prediction z_pred
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

    # Per-source normalization: flux and flux_err are scaled by the same factor
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
                   title="Supervised MLP Sample"):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    z_true = z_true.flatten(); z_pred = z_pred.flatten()
    sigma_nmad, eta, bias = compute_metrics(z_true, z_pred)

    ax.scatter(z_true, z_pred, s=8, alpha=0.3, color='royalblue',
               edgecolors='none', label='Galaxies')
    line_grid = np.linspace(z_min, z_max, 500)
    ax.plot(line_grid, line_grid, 'r--', lw=1.5, label=r'$z_{\mathrm{pred}} = z_{\mathrm{true}}$')
    ax.plot(line_grid, line_grid + 0.15 * (1 + line_grid), 'r:', lw=1.2, label=r'$\pm 0.15(1+z)$')
    ax.plot(line_grid, line_grid - 0.15 * (1 + line_grid), 'r:', lw=1.2)

    ax.set_xlim(z_min, z_max); ax.set_ylim(z_min, z_max)
    ax.set_xlabel(r'$z_{\mathrm{true}}$', fontsize=13)
    ax.set_ylabel(r'$z_{\mathrm{pred}}$', fontsize=13)
    ax.set_title(f"Supervised MLP Baseline  {title}", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    text_box = (f"$\\sigma_{{\\mathrm{{NMAD}}}} = {sigma_nmad:.4f}$\n"
                f"$\\eta (|\\Delta z|>0.15) = {eta:.2f}\\%$\n"
                f"$\\langle\\Delta z\\rangle = {bias:.4f}$")
    ax.text(0.05, 0.78, text_box, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.85, edgecolor='gray'))

    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_training_curve(history, best_epoch=None, save_path=CURVE_PNG):
    train_loss = np.array(history["train_loss"], dtype=np.float64)
    val_loss   = np.array(history["val_loss"], dtype=np.float64)
    ep = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(ep, train_loss, label="Train Supervised Loss (Huber)", lw=1.3)
    ax.plot(ep, val_loss,   label="Val   Supervised Loss (Huber)", lw=1.3)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", ls="--", lw=1.2,
                   label=f"Best epoch = {best_epoch}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (log)", fontsize=12)
    ax.set_title("Supervised Training / Validation Curve", fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


# =====================================================================
# 3. Supervised training pipeline: spectroscopic training set -> early stopping on spectroscopic validation set -> evaluation on photometric test set
# =====================================================================
def train_and_evaluate():
    print("\n==================================================")
    print(" Run mode: [Pure supervised MLP model + spectroscopic-label supervision + early stopping]")
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
    ds_train = (tf.data.Dataset.from_tensor_slices((f_train, e_train, z_train))
                .shuffle(min(10000, len(f_train)), seed=SEED)
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    ds_val   = (tf.data.Dataset.from_tensor_slices((f_val, e_val, z_val))
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    # ---- (C) Model + optimizer + supervised loss function ----
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

    def evaluate_dataset_loss(dataset, step_fn):
        total, count = 0.0, 0
        for f_b, e_b, z_b in dataset:
            bs = int(tf.shape(f_b)[0])
            total += float(step_fn(f_b, e_b, z_b)) * bs
            count += bs
        return total / max(count, 1)

    # ---- (D) Training loop + early stopping ----
    best_val   = np.inf
    best_epoch = 0
    wait       = 0
    history    = {"train_loss": [], "val_loss": []}
    nmad_records = []

    print(f"Starting supervised training (EarlyStopping patience={EARLY_STOP_PATIENCE}, "
          f"monitor=val supervised loss)...\n")

    for epoch in range(epochs):
        # train
        t_loss, steps = 0.0, 0
        for f_b, e_b, z_b in ds_train:
            t_loss += float(train_step(f_b, e_b, z_b)); steps += 1
        t_loss /= max(steps, 1)

        # validation
        v_loss = evaluate_dataset_loss(ds_val, val_step)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        # early stopping
        improved = v_loss < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val   = v_loss
            best_epoch = epoch + 1
            wait       = 0
            model.save_weights(CKPT_BEST_WEIGHTS)
        else:
            wait += 1

        if epoch == 0 or (epoch + 1) % 10 == 0:
            star = " *" if improved else ""
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {t_loss:.6f} | "
                  f"Val Loss: {v_loss:.6f} | Best: {best_val:.6f}@{best_epoch}{star}")

        # Diagnose validation NMAD every 20 epochs
        if epoch == 0 or (epoch + 1) % 20 == 0:
            z_vp = model.predict([f_val, e_val], batch_size=512, verbose=0)
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
        nmad_records=np.array(nmad_records, dtype=np.float32),
        best_epoch=np.int32(best_epoch),
        best_val_loss=np.float32(best_val),
    )
    plot_training_curve(history, best_epoch=best_epoch, save_path=CURVE_PNG)

    # ---- (G) Evaluate on training set, validation set, and photometric test set ----
    print("\n------------------ Evaluation results ------------------")
    z_pred_train = model.predict([f_train, e_train], batch_size=512, verbose=0)
    s_tr, eta_tr, b_tr = compute_metrics(z_train, z_pred_train)
    print("\n[Train set / spectroscopic sample]:")
    print(f"  * σ_NMAD             : {s_tr:.4f}")
    print(f"  * Outlier Fraction   : {eta_tr:.2f}%")
    print(f"  * Median Bias (<Δz>) : {b_tr:.4f}")

    z_pred_val = model.predict([f_val, e_val], batch_size=512, verbose=0)
    s_va, eta_va, b_va = compute_metrics(z_val, z_pred_val)
    print("\n[Validation set / spectroscopic sample]:")
    print(f"  * σ_NMAD             : {s_va:.4f}")
    print(f"  * Outlier Fraction   : {eta_va:.2f}%")
    print(f"  * Median Bias (<Δz>) : {b_va:.4f}")
    plot_z_scatter(z_val, z_pred_val, save_path=VALID_FIG_PNG,
                   title="Spectroscopic Validation Sample")

    # Photometric test set (to examine the domain shift / generalization collapse of the supervised model)
    z_pred_test = model.predict([f_test, e_test], batch_size=512, verbose=0)
    s_te, eta_te, b_te = compute_metrics(z_test, z_pred_test)
    print("\n[Test set / photometric sample evaluation ★]:")
    print(f"  * σ_NMAD             : {s_te:.4f}")
    print(f"  * Outlier Fraction   : {eta_te:.2f}%")
    print(f"  * Median Bias (<Δz>) : {b_te:.4f}")
    plot_z_scatter(z_test, z_pred_test, save_path=TEST_FIG_PNG,
                   title="Photometric Test Sample (Domain Shift Test)")

    # ---- (H) Save final weights ----
    model.save_weights(CKPT_FINAL_WEIGHTS)
    print("\nAll output files have been saved.")


if __name__ == "__main__":
    train_and_evaluate()
