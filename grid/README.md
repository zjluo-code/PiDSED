# make_grid: FSPS 11-band photometric template grid

This directory combines 12 EAZY/FSPS QSF v3 rest-frame templates, G23 dust attenuation, Inoue14 IGM absorption and 11 COSMOS filter response curves into a four-dimensional photometric grid that can be used directly for photometric-redshift fitting, template synthesis and stellar-population parameter scans. The grid shape is `401 × 50 × 12 × 11` (redshift × E(B-V) × template × band), 240,600 nodes in total; flux is in μJy (AB zeropoint 23.9) and normalized to AB = 25 mag in the reference band `UVISTA_Ks`, so every node has exactly `10**(-0.4*(25-23.9))` = 0.363078 μJy at Ks, and a fit only needs one additional free amplitude.

> **Path convention**: below, `code/make_grid/` denotes this directory inside the COSMOS project. If you use the contents of this directory directly as the root of a GitHub repository (with `make_grid.py`, `templates/`, `filters/`, `README.md` and `demo.ipynb` at the top level), drop that prefix — `python make_grid.py` works from the repository root, all inputs live inside this directory, and nothing else in the project is required.

## 1. Grid overview

| Axis | Size | Range | Meaning |
| --- | --- | --- | --- |
| Redshift `z` | 401 | 0 – 4, equally spaced, step 0.01 | Source redshift; also drives IGM absorption and the redshifted filters |
| Color excess `E(B-V)` | 50 | 0 – 1, equally spaced (step 1/49 ≈ 0.0204) | Color excess of the G23 dust attenuation; one template is scanned along this axis |
| Template | 12 | `tweak_fsps_QSF_12_v3_001` … `_012` | FSPS QSF v3 rest-frame SED (shape) |
| Band | 11 | `GALEX_NUV`, `CFHT_ustar`, `HSC_g`, `HSC_r`, `HSC_i`, `HSC_z`, `HSC_y`, `UVISTA_J`, `UVISTA_Ks`, `IRAC_CH1`, `IRAC_CH2` | Filter response curves; this order is the order of the last `flux` dimension, Ks has index 8 |

- Units and normalization: flux is in μJy (AB magnitude zeropoint 23.9); `F_b / F_Ks * 10**(-0.4*(25-23.9))`, i.e. the `UVISTA_Ks` flux of every node is set to 0.363078 μJy while template shape, dust colors and IGM absorption are all preserved; the absolute amplitude (distance, luminosity factor, overall template normalization) cancels out.
- Flux is always non-negative; because the amplitude is normalized away, the grid describes colors/relative fluxes only, not "how bright the source is".

## 2. Physics and formulas

For each node `(z, E, t)` and band `b`:

$$
F_b(z,E,t) \;=\; A \cdot
\frac{\displaystyle\int f_{\lambda,t}(\lambda)\,10^{-0.4\,k(\lambda)E}\,T_{\rm IGM}(\lambda,z)\,\lambda\,R_b((1+z)\lambda)\,\mathrm{d}\lambda}
{\displaystyle\int R_b((1+z)\lambda)\,\mathrm{d}\lambda/\lambda}
$$

where `f_{\lambda,t}` is the rest-frame template flux, `k(\lambda)` the dust attenuation curve, `T_{\rm IGM}` the IGM transmission, `R_b` the response of filter `b`, and `A` a free amplitude (distance, luminosity factor and absolute template calibration are absorbed by `A`; the grid itself fixes `A` to "reference band AB = 25 mag").

- **Dust**: the script uses `k(\lambda) = A(\lambda)/E(B-V) = R_v \cdot \mathrm{G23}(\lambda)` with `R_v = 3.1`. `dust_extinction`'s G23 returns the V-normalized `A(\lambda)/A(V)` (≈ 0.9985 at 0.55 μm); multiplying by `R_v = A(V)/E(B-V)` gives exactly the `A(\lambda)/E(B-V)` needed by the attenuation factor `10^{-0.4kE}` (e.g. `k(5500 Å)=3.095`, `k(4400 Å)=4.093`). The G23 domain is [912 Å, 32 μm]; the script clips the wavelength to `[912.0, 320000.0]` Å before evaluation and extends the endpoint values outside the domain — an approximation at the far-UV/IR ends, and an established convention shared with the notebook.
- **IGM**: for rest-frame `\lambda \geq 1300` Å, `T_{\rm IGM}=1`; absorption is computed only for the LyC region `\lambda < 1300` Å, calling `eazy.igm.Inoue14.full_IGM(z, lambda_rest*(1+z))` with the observed-frame wavelength. Transmission is 1 at z = 0.
- **Filters**: response curves come from lephare-data (two columns: wavelength [Å], response); on load, negative responses are clipped to 0 and the curve is normalized to unit peak. `R_b` appears in both numerator and denominator, so peak normalization does not affect the ratio; the response is evaluated at observed wavelength `(1+z)\lambda`, i.e. "the filter moves with redshift".
- **Integration**: the wavelength variable is rest-frame `\lambda`, integrated with the trapezoidal rule on a non-uniform grid; the denominator `\int R_b\,\mathrm{d}\lambda/\lambda` makes the result a weighted average SED with weights `R_b((1+z)\lambda)/\lambda`, independent of the absolute calibration of the response curve.
- **Wavelength grid**: `LAM = unique([geomspace(lam_lo, lam_hi, n_wave), template native nodes, 912, 1026, 1216])`, where `lam_lo = min(filter blue end)/(1+z_max)*0.99` and `lam_hi = max(filter red end)*1.01`; the default `n_wave=16000` gives 21043 points (332.64 – 53762.3 Å). 912/1026/1216 Å correspond to the LyC edge, Lyβ and Lyα, ensuring the IGM absorption edges are not under-sampled; all native template nodes are kept, avoiding interpolation smoothing of narrow absorption lines.
- **Normalization**: after synthesis the script applies `flux / flux[..., Ks] * 10**(-0.4*(ref_mag-23.9))`, i.e. it scales the reference-band flux of every node to a constant; `ref_mag` can be changed with `--ref-mag`.

## 3. Directory structure

```text
code/make_grid/
├── make_grid.py                            # standalone script + importable module (the whole implementation)
├── requirements.txt                        # runtime dependencies (with measured versions)
├── .gitignore                              # ignores output/, __pycache__ and similar
├── README.md / demo.ipynb                  # this file / demonstration notebook
├── templates/fsps_full/
│   ├── tweak_fsps_QSF_12_v3.param          # template manifest: one template path and weights per line
│   └── tweak_fsps_QSF_12_v3_001..012.dat   # 12 rest-frame SEDs, two columns (λ[Å], f_λ)
├── filters/
│   ├── manifest.json                       # provenance and sha256 of the 40 lephare-data filters
│   └── filt/{cosmos,hsc,vista}/            # verbatim copies of the upstream filter files
└── output/                                 # generated at runtime, ignored by .gitignore
    ├── fsps_11band_grid.npz
    ├── grid_metadata.json
    └── grid_diagnostics.png
```

- `make_grid.py`: the only implementation, covering template/filter loading with sha256 verification, wavelength-grid construction, dust and IGM synthesis, normalization, content hashing/caching, the diagnostics figure and the command-line entry point.
- `templates/fsps_full/`: the `.param` is an EAZY-style manifest whose lines look like `1 templates/fsps_full/tweak_fsps_QSF_12_v3_001.dat 1.0 0.0 1.0`; each `.dat` has 5994 rows and two columns (λ[Å], f_λ), without a header.
- `filters/manifest.json`: records the repository URL, commit, and for every filter its `profile` relative path, `sha256`, `source_url` and more; 40 filters are registered, by default only 11 are used.
- `filters/filt/`: verbatim copies of the upstream filter files (`.pb` / `.lowres` / `.res`), keeping the upstream header comments and the README of each subdirectory.
- `output/`: generated at runtime and not committed; the `11` in the file name comes from the number of bands (`fsps_{n_band}band_grid.npz`).

## 4. Installation

Python >= 3.10 is required (the source uses 3.10+ syntax such as `str | Path`).

```bash
pip install -r requirements.txt        # preferably inside a venv
```

| Package | Requirement | Purpose |
| --- | --- | --- |
| `numpy` | >= 1.24 | arrays and numerical integration |
| `scipy` | >= 1.10 | a dependency of `eazy` |
| `astropy` | >= 5.3 | Quantity input for `dust_extinction` |
| `matplotlib` | >= 3.7 | only the diagnostics figure `grid_diagnostics.png` |
| `dust_extinction` | >= 1.2 | G23 dust attenuation curve |
| `eazy-py` | >= 0.7 | Inoue14 IGM (imported as `eazy`) |
| `threadpoolctl` | >= 3.1 | limit the BLAS/OpenMP thread count (optional) |

Environment used here: Python 3.15.1, numpy 2.5.2, scipy 1.18.1, astropy 8.0.1, eazy 0.8.7, dust_extinction 1.7, matplotlib 3.11.2, threadpoolctl 3.6.0.

## 5. Quick start

```bash
cd code/make_grid
python make_grid.py --help     # show all options
python make_grid.py            # build the default 401 × 50 × 12 × 11 grid
```

Input directories are resolved relative to the script location, so it can also run from any directory: `python code/make_grid/make_grid.py`.

Demo notebook (with all outputs stored): `jupyter lab demo.ipynb`, or open `demo.ipynb` in VS Code. Its order is: build the grid → 11-band fluxes and filter curves of a single node → amplitude-marginalized χ² fit → custom configuration and wavelength-convergence check → save/load and diagnostics → CLI/notebook bit-identical check; it runs in about 10 s, and when this directory is used standalone the final "original notebook artifact comparison" is skipped automatically.

Measured timing (macOS arm64, `--threads 2`, Python 3.15.1): full synthesis 2.1–2.3 s, end-to-end about 3.4–4.0 s; wavelength grid 21043 points, `fsps_11band_grid.npz` about 18.8 MB. Shrinking `--n-z`/`--n-ebv` finishes in under a second (e.g. 5 × 50 about 0.1 s); `--n-wave 32000` gives 37043 wavelength points and 4.4 s of synthesis.

Artifacts in `output/`:

| File | Content |
| --- | --- |
| `fsps_11band_grid.npz` | compressed npz, keys listed below |
| `grid_metadata.json` | indented JSON of the same `metadata` dictionary (`sort_keys=True`, `ensure_ascii=False`) |
| `grid_diagnostics.png` | left: band flux over Ks flux ratios of template 001 at E(B-V)=0 versus redshift; right: the 12 rest-frame template SEDs, each normalized at 5500 Å |

Keys of the npz (readable directly with `np.load(..., allow_pickle=False)`):

| Key | dtype / shape | Meaning |
| --- | --- | --- |
| `flux` | float64 `(401, 50, 12, 11)` | photometric grid, μJy, axis order (z, E(B-V), template, band) |
| `redshift` | float64 `(401,)` | redshift grid |
| `ebv` | float64 `(50,)` | E(B-V) grid |
| `template_names` | `<U24` `(12,)` | template names (e.g. `tweak_fsps_QSF_12_v3_004`) |
| `filter_names` | `<U10` `(11,)` | band names, same order as the last `flux` dimension |
| `ref_filter` | `<U9` scalar | normalization reference band, default `UVISTA_Ks` |
| `ref_mag` | float64 scalar | AB magnitude of the reference band, default 25.0 |
| `config_hash` | `<U64` scalar | content hash (see section 9) |
| `metadata` | JSON string scalar | full metadata, identical to `grid_metadata.json` |

Top-level keys of `grid_metadata.json`: `config`, `data_provenance`, `elapsed_seconds`, `flux_statistics`, `flux_unit`, `format_version`, `generator`, `inputs`, `n_nodes`, `reference_flux_uJy`, `shape`, `software`, `version`, `wavelength_grid`.

If the output directory already contains an npz with the same `config_hash`, a repeated run hits the cache: synthesis is skipped and only the metadata and diagnostics figure are rewritten; use `--force` to force recomputation.

## 6. Command-line options

**Grid definition**

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--bands BAND [BAND ...]` | list[str] | the 11 default bands | bands to synthesize; the order sets the grid's last axis; each must appear in `manifest.json` |
| `--z-min` | float | 0.0 | minimum redshift |
| `--z-max` | float | 4.0 | maximum redshift |
| `--n-z` | int | 401 | number of redshift grid points |
| `--ebv-min` | float | 0.0 | lower E(B-V) limit |
| `--ebv-max` | float | 1.0 | upper E(B-V) limit |
| `--n-ebv` | int | 50 | number of E(B-V) grid points |
| `--rv` | float | 3.1 | `R_v` of the G23 curve |
| `--ref-band` | str | `UVISTA_Ks` | reference band used for normalization |
| `--ref-mag` | float | 25.0 | AB magnitude of the reference band |
| `--n-wave` | int | 16000 | dense log-spaced rest-frame wavelength points (native template nodes are added) |

**Inputs**

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--base-dir` | path | directory containing the script | base directory for relative paths |
| `--template-param` | path | `templates/fsps_full/tweak_fsps_QSF_12_v3.param` | template `.param` file, resolved relative to `--base-dir` |
| `--filter-dir` | path | `filters` | root directory of the filters |
| `--manifest` | path | `filters/manifest.json` | filter manifest |
| `--no-verify` | flag | verification is on by default | skip the filter sha256 verification (faster, but not recommended) |

**Output and execution**

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--output-dir` | path | `<base-dir>/output` | output directory; a relative path is resolved against the current working directory |
| `--threads` | int | 2 | cap on BLAS/OpenMP threads |
| `--progress-every` | int | 50 | print progress every N redshift steps (0 = silent) |
| `--force` | flag | off | ignore the content-hash cache and recompute |
| `--no-diagnostics` | flag | diagnostics on by default | do not produce the diagnostics figure |
| `--quiet` | flag | off | print no log output at all |
| `--version` | flag | — | print the version and exit |
| `-h, --help` | flag | — | print help and exit |

Parameter constraints: `n_z >= 2`, `n_ebv >= 1`, `n_wave >= 1000`, `rv > 0`, `z_max > z_min`, `ebv_max >= ebv_min`, `ref_band` must be included in `--bands`, and `--bands` must be neither empty nor contain duplicates. Any invalid input is printed to stderr as `error: ...` and the process exits with status 1.

## 7. Use as a library

`make_grid.py` is also an importable module (no installation needed); the examples below assume the current directory is the repository root. Build a small grid and save it:

```python
import sys
sys.path.insert(0, "code/make_grid")

from make_grid import GridConfig, build_grid, save_grid, load_grid

cfg = GridConfig(z_max=2.0, n_z=201, n_ebv=20)   # other fields keep their defaults
grid = build_grid(cfg)                            # returns a TemplateGrid
print(grid.shape, grid.n_nodes, grid.config_hash[:12])
save_grid(grid, "code/make_grid/output/demo")

# load a saved grid back
grid = load_grid("code/make_grid/output/fsps_11band_grid.npz")
print(grid.shape, grid.redshift[0], grid.redshift[-1])   # (401, 50, 12, 11) 0.0 4.0
print(grid.metadata["wavelength_grid"]["n"])             # 21043
```

Photometric-redshift fitting with amplitude marginalization on the saved grid (`a = \max(0, \sum f m/\sigma^2 \,/\, \sum m^2/\sigma^2)`, `\chi^2 = \sum (f - a\,m)^2/\sigma^2`):

```python
import numpy as np

grid = load_grid("code/make_grid/output/fsps_11band_grid.npz")
models = grid.flux.reshape(-1, grid.flux.shape[-1])      # (240600, 11), μJy

# mock source at z=1.37, template 004, AB=24.5 mag
f_true = grid.flux[137, 0, 3] * 10 ** (-0.4 * (24.5 - 25.0))
sigma = 0.05 * f_true
f_obs = f_true + np.random.default_rng(0).normal(0.0, sigma)

w = 1.0 / sigma ** 2
a = np.maximum(0.0, (models * w) @ f_obs / ((models ** 2 * w).sum(axis=1)))
chi2 = (((f_obs - a[:, None] * models) ** 2) * w).sum(axis=1)
best = int(chi2.argmin())
iz, ie, it = np.unravel_index(best, grid.flux.shape[:-1])
print(f"z={grid.redshift[iz]:.2f}  E(B-V)={grid.ebv[ie]:.3f}  "
      f"{grid.template_names[it]}  A={a[best]:.3f}")
```

The amplitude-marginalized `\chi^2` is invariant under an overall scaling, so the source flux does not need to be normalized to Ks = 25 mag first; it is enough that `f_obs`, `models` and `sigma` use the same flux unit (μJy).

## 8. Data sources and licensing

**Templates (EAZY / FSPS)**: the 12 rest-frame SEDs come from the FSPS QSF v3 template set `tweak_fsps_QSF_12_v3` of EAZY (eazy-py); the `.param` manifest and the `.dat` files are stored verbatim in the upstream format. eazy-py is released under the MIT license; please cite Conroy et al. (2009) and Conroy & Gunn (2010) for FSPS, and Brammer et al. (2008) for EAZY.

**Filters (lephare-data)**: source repository <https://github.com/lephare-photoz/lephare-data>, commit `4d4dabf4f3ffda63e0d93c1d01956564485101e4`. `filters/manifest.json` records for every filter its `profile` (relative path), `sha256` and `source_url`; at runtime each sha256 is recomputed and compared with the manifest, and any mismatch is an error (`--no-verify` skips this, not recommended).

Filter files are copied verbatim from upstream, keeping the header comments and the README of each subdirectory; negative-value clipping and peak normalization happen only at runtime, the files on disk are not modified. The lephare-data repository root has no unified LICENSE, and the copyright of the filter response curves belongs to the respective instrument teams — check the upstream terms before redistribution or commercial use. When using LePHARE data please cite Arnouts et al. (1999) and Ilbert et al. (2006).

**Other dependencies**: the G23 dust curve comes from `dust_extinction` (Gordon et al. 2023, BSD-3-Clause); the Inoue14 IGM is provided by `eazy.igm` (Inoue et al. 2014); numpy, scipy, astropy and threadpoolctl are under BSD-style licenses, matplotlib uses a PSF-style license.

## 9. Reproducibility and checks

**Content hash and cache.** `config_hash` is the sha256 of: the grid format version (2), the full `GridConfig` (band list, z/E(B-V) ranges and counts, `R_v`, reference band and magnitude, `n_wave`, the AB zeropoint 23.9), the IGM implementation marker (`eazy.igm.Inoue14`), the dust clipping domain [912, 320000] Å, the sha256 of the wavelength-grid array bytes, the sha256 of every template and filter file involved (recorded by absolute path), the source text of the four synthesis functions (`dust_k`, `dust_seds`, `synth_flux`, `normalize`), and the versions of numpy/astropy/eazy/dust_extinction/threadpoolctl. The hash is written into the npz and the metadata; rerunning with the same `--output-dir` reuses the cache when the hash matches (`--force` forces recomputation regardless). Note that the hash includes absolute paths and function source text (including docstrings/comments), so moving the directory or editing comments in these functions also invalidates the cache — the price of "same hash means trustworthy", and expected behavior.

**Runtime assertions.**

- Template loading: wavelengths strictly increasing, SED finite and non-negative.
- Filter loading: band present in the manifest, file exists, sha256 matches the manifest; negative responses are clipped to 0 and the curve normalized to peak 1.
- Wavelength coverage: every template must cover `[lam_lo, lam_hi]`; during synthesis an integration denominator `<= 0` raises immediately.
- Normalization: the reference-band flux must be finite and positive; the final grid must be entirely finite and non-negative (otherwise `ValueError: grid contains non-finite or negative flux`), and the reference band must agree with `reference_flux_uJy(ref_mag)` (otherwise `reference-band normalization failed`). A default run prints a reference flux of 0.363078 μJy at the end.

**Numerical integration accuracy.** Cell 10 of the original notebook compares a coarse grid with a midpoint-refined grid over 6 redshifts and 3 E(B-V) values, checking `|\Delta \mathrm{mag}|` against a threshold of 0.01 mag. To keep the scope of "extracting only the grid-construction part", this package does not include that check, but it can be reproduced manually by doubling `--n-wave`:

```bash
cd code/make_grid
python3 make_grid.py --output-dir /tmp/grid_16k --no-diagnostics
python3 make_grid.py --output-dir /tmp/grid_32k --no-diagnostics --n-wave 32000
python3 - <<'PY'
import numpy as np
a = np.load("/tmp/grid_16k/fsps_11band_grid.npz")["flux"]
b = np.load("/tmp/grid_32k/fsps_11band_grid.npz")["flux"]
ks = 8                                   # index of UVISTA_Ks in the default band order
use = (b > 0) & (b / b[..., ks, None] > 1e-8)
dm = np.abs(2.5 * np.log10(b[use] / a[use]))
print(len(dm), dm.max(), np.median(dm))
PY
```

Measured (default 16000 vs 32000 points): about 2.54×10⁶ elements compared, `max|Δmag| = 0.00645`, median 1.9e-6, below 0.01 mag; the two grids are not bit-identical (different integration points are expected), the difference is only meaningful as `|\Delta \mathrm{mag}|`. The same refinement comparison is also in section 4 of `demo.ipynb` (fixed 21 × 5 axes, `n_wave` 2000 vs 8000, `max|Δmag|` = 1.5e-3).

**Bit-identical to the notebook.** The `flux` in `output/fsps_11band_grid.npz` satisfies `np.array_equal` with the original notebook output = `True`; the sha256 of the flux bytes is `9083d363bfe3b79a6582c38b975fe917b2856bb0807f9985113df08b8c23fd4f`.

This can be re-checked with:

```bash
cd code/make_grid
python3 - <<'PY'
import hashlib, numpy as np
flux = np.load("output/fsps_11band_grid.npz")["flux"]
print(hashlib.sha256(np.ascontiguousarray(flux).tobytes()).hexdigest())
PY
```
