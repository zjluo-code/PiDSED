#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_grid -- FSPS templates + dust attenuation + IGM absorption photometric grid.

Extracted from section 3 of cosmos_fsps_template_grid_11band.ipynb (the FSPS + dust + IGM
grid). The numerics stay bit-for-bit identical to the notebook, but the script no longer
depends on the notebook, on the COSMOS catalog or on any path inside that project.

Physical model
--------------
For every redshift z, extra attenuation E(B-V) and template spectrum f_lambda:

    F_b  ∝  ∫ f_lambda(λ) · 10^{-0.4 k(λ) E(B-V)} · T_IGM(λ, z) · λ R_b((1+z)λ) dλ
            /  ∫ R_b((1+z)λ) dλ / λ

The band fluxes are synthesized in the observed frame with a non-uniform trapezoidal rule
and then normalized to 25 AB in a reference band (Ks by default). The common luminosity
distance factor and the (1+z) scaling are absorbed by a free amplitude, so the grid only
carries relative flux shapes. Zero flux stays 0 and is never turned into a fake magnitude.

Usage
----
    python make_grid.py                          # default 401x50x12x11 grid (a few seconds)
    python make_grid.py --n-z 51 --n-ebv 10      # quick smoke test while debugging
    python make_grid.py --bands HSC_g HSC_r HSC_i --ref-band HSC_i
    python make_grid.py --help

Output (output/ by default)
--------------------------
    output/fsps_11band_grid.npz    flux (401, 50, 12, 11) plus axes, template/band names, config_hash
    output/grid_metadata.json      configuration, input sha256, software versions, statistics
    output/grid_diagnostics.png    band flux versus redshift + rest-frame spectra of the templates

As a library
------------
    from make_grid import GridConfig, build_grid, save_grid, load_grid

    config = GridConfig(n_z=51)
    grid = build_grid(config)          # TemplateGrid
    save_grid(grid, "output/")

See README.md for data provenance; templates and filters live in templates/ and filters/.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import sys
import time
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import astropy.units as u
from dust_extinction.parameter_averages import G23

try:  # IGM transmission: EAZY's Inoue14 implementation is used by default
    from eazy.igm import Inoue14
except ImportError:  # pragma: no cover - main() turns this into a readable error
    Inoue14 = None

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional dependency
    threadpool_limits = None

__version__ = "1.0.0"

# --------------------------------------------------------------------------------------
# Default configuration
# --------------------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent

#: The 11 COSMOS2020 bands; this order is the order of the grid's last axis
DEFAULT_BANDS = (
    "GALEX_NUV",
    "CFHT_ustar",
    "HSC_g",
    "HSC_r",
    "HSC_i",
    "HSC_z",
    "HSC_y",
    "UVISTA_J",
    "UVISTA_Ks",
    "IRAC_CH1",
    "IRAC_CH2",
)

DEFAULT_TEMPLATE_PARAM = Path("templates") / "fsps_full" / "tweak_fsps_QSF_12_v3.param"
DEFAULT_FILTER_DIRNAME = "filters"
DEFAULT_MANIFEST_NAME = "manifest.json"
DEFAULT_OUTPUT_DIRNAME = "output"

#: AB magnitude 23.9 in microjansky (EAZY's PRIOR_ABZP); hence the uJy flux unit
AB_ZP_UJY = 23.9

#: Domain of the G23 curve (912 A - 32 um); outside it the value is clamped to the edge
DUST_LAM_MIN = 912.0
DUST_LAM_MAX = 320000.0

#: IGM absorption is computed only for rest-frame wavelengths below 1300 A (Inoue14 LyC)
IGM_LAM_MAX = 1300.0

#: Rest-frame spectral features always kept on the wavelength grid (A)
FEATURE_WAVELENGTHS = (912.0, 1026.0, 1216.0)

#: Grid file format version; it takes part in the content hash
GRID_FORMAT_VERSION = 2


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridConfig:
    """Describe the whole grid at once. Every field takes part in config_hash."""

    bands: tuple[str, ...] = DEFAULT_BANDS
    z_min: float = 0.0
    z_max: float = 4.0
    n_z: int = 401
    ebv_min: float = 0.0
    ebv_max: float = 1.0
    n_ebv: int = 50
    rv: float = 3.1
    ref_band: str = "UVISTA_Ks"
    ref_mag: float = 25.0
    n_wave: int = 16000

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", tuple(self.bands))
        if len(self.bands) == 0:
            raise ValueError("bands must not be empty")
        if len(set(self.bands)) != len(self.bands):
            raise ValueError(f"duplicate bands: {self.bands}")
        if self.n_z < 2:
            raise ValueError(f"n_z must be at least 2, got {self.n_z}")
        if self.n_ebv < 1:
            raise ValueError(f"n_ebv must be at least 1, got {self.n_ebv}")
        if not self.z_max > self.z_min:
            raise ValueError(f"need z_max > z_min, got {self.z_min}-{self.z_max}")
        if not self.ebv_max >= self.ebv_min:
            raise ValueError(f"need ebv_max >= ebv_min, got {self.ebv_min}-{self.ebv_max}")
        if self.rv <= 0:
            raise ValueError(f"rv must be positive, got {self.rv}")
        if self.n_wave < 1000:
            raise ValueError(f"n_wave too small ({self.n_wave}); the dense grid needs at least 1000 points")
        if self.ref_band not in self.bands:
            raise ValueError(f"ref_band={self.ref_band!r} must appear in bands")

    @property
    def z_grid(self) -> np.ndarray:
        return np.linspace(self.z_min, self.z_max, self.n_z)

    @property
    def ebv_grid(self) -> np.ndarray:
        return np.linspace(self.ebv_min, self.ebv_max, self.n_ebv)

    def to_dict(self) -> dict:
        """Plain-JSON representation used for hashing and metadata."""
        return {
            "bands": list(self.bands),
            "z": {"min": self.z_min, "max": self.z_max, "n": self.n_z},
            "ebv": {"min": self.ebv_min, "max": self.ebv_max, "n": self.n_ebv},
            "rv": self.rv,
            "ref_band": self.ref_band,
            "ref_mag": self.ref_mag,
            "ab_zeropoint_uJy": AB_ZP_UJY,
            "n_wave": self.n_wave,
        }


# --------------------------------------------------------------------------------------
# Inputs: templates and filters
# --------------------------------------------------------------------------------------


@dataclass
class TemplateSet:
    """A set of rest-frame templates: names, files, wavelengths and f_lambda."""

    names: list[str]
    paths: list[Path]
    wave: list[np.ndarray]
    sed: list[np.ndarray]
    param_file: Path

    def __len__(self) -> int:
        return len(self.names)


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 (used for template/filter checks and the content hash)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_template_path(rel: str, param_file: Path) -> Path:
    """Resolve a relative template path from the .param file to a real file.

    EAZY .param files list paths relative to templates/ (e.g.
    templates/fsps_full/x.dat); common layouts are tried by walking up from the param
    file, and the first hit wins.
    """
    rel_path = Path(rel)
    candidates = [rel_path] if rel_path.is_absolute() else []
    candidates += [
        param_file.parent / rel_path,
        param_file.parent.parent / rel_path,
        param_file.parent.parent.parent / rel_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"template file {rel!r} not found (param: {param_file}); tried: "
        + ", ".join(str(c) for c in candidates)
    )


def load_templates(param_file: str | Path) -> TemplateSet:
    """Read an EAZY-style .param file and the template spectra it points to.

    Each line looks like <index> <relative path> <amplitude> <wavelength range...>;
    blank lines and lines starting with # are ignored.
    """
    param_file = Path(param_file).expanduser().resolve()
    if not param_file.is_file():
        raise FileNotFoundError(f"template .param file does not exist: {param_file}")

    paths: list[Path] = []
    for line in param_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        paths.append(_resolve_template_path(fields[1], param_file))
    if not paths:
        raise ValueError(f"no template lines parsed from {param_file}")

    names, waves, seds = [], [], []
    for path in paths:
        wave, sed = np.loadtxt(path, usecols=(0, 1)).T
        wave = np.asarray(wave, dtype=float)
        sed = np.asarray(sed, dtype=float)
        if not np.all(np.diff(wave) > 0):
            raise ValueError(f"{path.name}: wavelength is not strictly increasing")
        if not (np.all(np.isfinite(sed)) and np.all(sed >= 0)):
            raise ValueError(f"{path.name}: f_lambda must be finite and non-negative")
        names.append(path.stem)
        waves.append(wave)
        seds.append(sed)
    return TemplateSet(names=names, paths=paths, wave=waves, sed=seds, param_file=param_file)


def load_filters(
    filter_dir: str | Path,
    manifest_file: str | Path,
    bands: tuple[str, ...] | list[str],
    verify: bool = True,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict], dict]:
    """Read the band filter response curves named by the manifest.

    Each response is a two-column text file (wavelength in A, response). As in the
    notebook, negative responses are clipped to 0 and the curve is normalized to a peak
    of 1. With verify=True the sha256 of every file is checked against the manifest.
    """
    filter_dir = Path(filter_dir).expanduser().resolve()
    manifest_file = Path(manifest_file).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"filter manifest does not exist: {manifest_file}")

    manifest = json.loads(manifest_file.read_text())
    row_by_band = {row["band"]: row for row in manifest["filters"]}
    missing = [band for band in bands if band not in row_by_band]
    if missing:
        raise KeyError(
            f"bands missing from the manifest: {missing}; "
            f"{sorted(row_by_band)}"
        )

    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: dict[str, dict] = {}
    for band in bands:
        row = row_by_band[band]
        path = filter_dir / row["profile"]
        if not path.is_file():
            raise FileNotFoundError(f"filter response file missing: {path} (band {band})")
        if verify and row.get("sha256"):
            actual = sha256_file(path)
            if actual != row["sha256"]:
                raise ValueError(
                    f"sha256 of {path} does not match the manifest:\n  expected {row['sha256']}\n  actual   {actual}"
                )
        wave, response = np.loadtxt(path, usecols=(0, 1)).T
        wave = np.asarray(wave, dtype=float)
        response = np.maximum(np.asarray(response, dtype=float), 0.0)
        peak = response.max()
        if not peak > 0:
            raise ValueError(f"{path}: response is all zero")
        profiles[band] = (wave, response / peak)
        rows[band] = row
    return profiles, rows, manifest


def build_wavelength_grid(
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    template_waves: list[np.ndarray],
    z_max: float,
    n_wave: int = 16000,
) -> np.ndarray:
    """Build the rest-frame wavelength grid: dense log grid + native nodes + key features.

    The end points follow the filter coverage: the shortest wavelength must cover the
    blue end of the bluest band at z=z_max, the longest one the red end of the reddest
    band.
    """
    lam_lo = min(wave[0] for wave, _ in profiles.values()) / (1.0 + z_max) * 0.99
    lam_hi = max(wave[-1] for wave, _ in profiles.values()) * 1.01
    for index, wave in enumerate(template_waves):
        if not (wave[0] <= lam_lo and wave[-1] >= lam_hi):
            raise ValueError(
                f"template #{index + 1} covers {wave[0]:.1f}-{wave[-1]:.1f} A, "
                f"which does not include the required range {lam_lo:.1f}-{lam_hi:.1f} A"
            )
    native = np.unique(np.concatenate([w[(w >= lam_lo) & (w <= lam_hi)] for w in template_waves]))
    dense = np.geomspace(lam_lo, lam_hi, n_wave)
    return np.unique(np.r_[dense, native, np.asarray(FEATURE_WAVELENGTHS, dtype=float)])


# --------------------------------------------------------------------------------------
# Synthesis: dust, IGM, filter integration, normalization
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _g23(rv: float) -> G23:
    """G23 attenuation curve object, cached per Rv."""
    return G23(Rv=rv)


def dust_k(wave: np.ndarray, rv: float = 3.1) -> np.ndarray:
    """G23 attenuation coefficient k(lam) = A(lam)/E(B-V), clamped to [912 A, 32 um].

    Note: this is written Rv * G23(lam), exactly as in the notebook. G23 already
    returns A(lam)/E(B-V), so the extra factor Rv amplifies the effective attenuation.
    It is kept unchanged to preserve bit-for-bit agreement with the existing grid; see
    the README for a discussion.
    """
    attenuation = _g23(float(rv))(np.clip(wave, DUST_LAM_MIN, DUST_LAM_MAX) * u.AA)
    return float(rv) * np.asarray(attenuation, dtype=float)


def dust_seds(
    wave: np.ndarray,
    template_waves: list[np.ndarray],
    template_seds: list[np.ndarray],
    ebvs: np.ndarray,
    rv: float = 3.1,
) -> np.ndarray:
    """Interpolate each template onto the common grid and apply dust; returns (n_ebv, n_template, n_wave)."""
    base = np.array([np.interp(wave, w, s) for w, s in zip(template_waves, template_seds)])
    k = dust_k(wave, rv)
    return base[None, :, :] * 10.0 ** (
        -0.4 * np.asarray(ebvs, dtype=float)[:, None, None] * k[None, None, :]
    )


def synth_flux(
    seds: np.ndarray,
    z: float,
    wave: np.ndarray,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    bands: tuple[str, ...] | list[str],
    igm,
) -> np.ndarray:
    """Synthesize band fluxes at redshift z; shape is seds.shape[:-1] + (len(bands),).

    Integral (observed frame, non-uniform trapezoid): numerator ~ int f_lam lam R_b((1+z)lam)
    T_IGM dlam; denominator ~ int R_b dlam / lam.
    """
    transmission = np.ones_like(wave)
    use = wave < IGM_LAM_MAX
    if z > 0:
        transmission[use] = igm.full_IGM(z, wave[use] * (1.0 + z))

    one_plus_z = 1.0 + z
    result = np.empty((*seds.shape[:-1], len(bands)), dtype=float)
    for j, band in enumerate(bands):
        filter_wave, filter_response = profiles[band]
        lo = max(0, int(np.searchsorted(wave, filter_wave[0] / one_plus_z)) - 1)
        hi = min(len(wave), int(np.searchsorted(wave, filter_wave[-1] / one_plus_z)) + 1)
        x = wave[lo:hi]
        response = np.interp(x * one_plus_z, filter_wave, filter_response, left=0.0, right=0.0)
        dx = np.diff(x)
        weights = np.r_[dx[0] / 2.0, (dx[:-1] + dx[1:]) / 2.0, dx[-1] / 2.0]
        denominator = np.sum(weights * response / x)
        if not denominator > 0:
            raise ValueError(f"filter integral of band {band} is 0 at z={z}; check the response coverage")
        numerator = weights * x * response * transmission[lo:hi]
        result[..., j] = (seds[..., lo:hi] @ numerator) / denominator
    return result


def normalize(
    flux: np.ndarray,
    bands: tuple[str, ...] | list[str],
    ref_band: str,
    ref_mag: float,
    ab_zeropoint: float = AB_ZP_UJY,
) -> np.ndarray:
    """Normalize to ref_mag AB with the reference band; the output is in uJy."""
    reference = flux[..., list(bands).index(ref_band)]
    if not (np.all(np.isfinite(reference)) and np.all(reference > 0)):
        raise ValueError(f"flux of reference band {ref_band} must be finite and positive")
    scale = 10.0 ** (-0.4 * (ref_mag - ab_zeropoint))
    return flux / reference[..., None] * scale


def reference_flux_uJy(ref_mag: float) -> float:
    """Flux (uJy) the reference band should have at ref_mag AB; used by self-checks and README examples."""
    return 10.0 ** (-0.4 * (ref_mag - AB_ZP_UJY))


# --------------------------------------------------------------------------------------
# Grid computation
# --------------------------------------------------------------------------------------


@dataclass
class TemplateGrid:
    """4-D (z, E(B-V), template, band) flux grid and its axes."""

    flux: np.ndarray
    redshift: np.ndarray
    ebv: np.ndarray
    template_names: list[str]
    filter_names: list[str]
    ref_filter: str
    ref_mag: float
    config_hash: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (
            len(self.redshift),
            len(self.ebv),
            len(self.template_names),
            len(self.filter_names),
        )
        if self.flux.shape != expected:
            raise ValueError(f"flux shape {self.flux.shape} does not match the axes {expected}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.flux.shape

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.flux.shape[:-1]))

    def npz_kwargs(self) -> dict:
        """npz key names identical to the ones the notebook saved."""
        return dict(
            flux=self.flux,
            redshift=self.redshift,
            ebv=self.ebv,
            template_names=np.asarray(self.template_names, dtype=str),
            filter_names=np.asarray(self.filter_names, dtype=str),
            ref_filter=self.ref_filter,
            ref_mag=np.float64(self.ref_mag),
            config_hash=self.config_hash,
            metadata=json.dumps(self.metadata, sort_keys=True),
        )


@dataclass
class Assets:
    """Every input loaded once, so nothing is read from disk twice."""

    config: GridConfig
    templates: TemplateSet
    profiles: dict[str, tuple[np.ndarray, np.ndarray]]
    filter_rows: dict[str, dict]
    manifest: dict
    manifest_file: Path
    filter_dir: Path
    wave: np.ndarray
    config_hash: str


def _software_versions() -> dict:
    """Software versions that take part in the content hash."""
    versions = {"numpy": np.__version__}
    try:
        import astropy

        versions["astropy"] = astropy.__version__
    except Exception:  # pragma: no cover
        pass
    for name in ("eazy", "dust_extinction", "threadpoolctl"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def compute_config_hash(
    config: GridConfig,
    templates: TemplateSet,
    manifest_file: Path,
    filter_dir: Path,
    wave: np.ndarray,
) -> str:
    """Content-addressed hash: config + all input files + wavelength grid + implementation source + software versions."""
    manifest = json.loads(Path(manifest_file).read_text())
    row_by_band = {row["band"]: row for row in manifest["filters"]}
    files = {str(path): sha256_file(path) for path in templates.paths}
    for band in config.bands:
        path = Path(filter_dir) / row_by_band[band]["profile"]
        files[str(path)] = sha256_file(path)
    payload = {
        "format_version": GRID_FORMAT_VERSION,
        "grid": config.to_dict(),
        "igm": "eazy.igm.Inoue14",
        "dust_domain_clamped": [DUST_LAM_MIN, DUST_LAM_MAX],
        "wave_sha256": hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest(),
        "files": files,
        "implementation": "".join(
            inspect.getsource(func) for func in (dust_k, dust_seds, synth_flux, normalize)
        ),
        "software": _software_versions(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_assets(
    config: GridConfig,
    *,
    base_dir: str | Path | None = None,
    template_param: str | Path | None = None,
    filter_dir: str | Path | None = None,
    manifest_file: str | Path | None = None,
    verify_filters: bool = True,
) -> Assets:
    """Load templates and filters, build the wavelength grid, return Assets carrying its config_hash."""
    base_dir = Path(base_dir).expanduser().resolve() if base_dir else PACKAGE_DIR

    def resolve(value, default) -> Path:
        if value is None:
            return (base_dir / default).resolve()
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()

    param_path = resolve(template_param, DEFAULT_TEMPLATE_PARAM)
    filters_root = resolve(filter_dir, DEFAULT_FILTER_DIRNAME)
    if manifest_file is None:
        manifest_path = filters_root / DEFAULT_MANIFEST_NAME
    else:
        manifest_path = resolve(manifest_file, DEFAULT_FILTER_DIRNAME / DEFAULT_MANIFEST_NAME)
        if not manifest_path.is_file() and (filters_root / manifest_path.name).is_file():
            manifest_path = filters_root / manifest_path.name

    templates = load_templates(param_path)
    profiles, rows, manifest = load_filters(
        filters_root, manifest_path, config.bands, verify=verify_filters
    )
    wave = build_wavelength_grid(profiles, templates.wave, config.z_max, config.n_wave)
    digest = compute_config_hash(config, templates, manifest_path, filters_root, wave)
    return Assets(
        config=config,
        templates=templates,
        profiles=profiles,
        filter_rows=rows,
        manifest=manifest,
        manifest_file=manifest_path,
        filter_dir=filters_root,
        wave=wave,
        config_hash=digest,
    )


def _make_igm():
    if Inoue14 is None:
        raise ImportError(
            "eazy must provide the Inoue14 IGM model: pip install eazy-py (see requirements.txt)"
        )
    return Inoue14()


def _synthesize(
    assets: Assets,
    *,
    threads: int = 2,
    progress_every: int = 50,
    log=print,
) -> TemplateGrid:
    """Core loop: synthesize the whole (E(B-V), template, band) block redshift by redshift."""
    config = assets.config
    z_grid = config.z_grid
    ebv_grid = config.ebv_grid
    bands = list(config.bands)

    sed_dust = dust_seds(
        assets.wave, assets.templates.wave, assets.templates.sed, ebv_grid, config.rv
    )
    flux = np.empty(
        (len(z_grid), len(ebv_grid), len(assets.templates), len(bands)), dtype=np.float64
    )
    igm = _make_igm()
    limiter = (
        threadpool_limits(limits=threads)
        if (threadpool_limits and threads)
        else contextlib.nullcontext()
    )

    started = time.monotonic()
    with limiter:
        for iz, z in enumerate(z_grid):
            flux[iz] = normalize(
                synth_flux(sed_dust, float(z), assets.wave, assets.profiles, bands, igm),
                bands,
                config.ref_band,
                config.ref_mag,
            )
            if progress_every and (iz % progress_every == 0 or iz == len(z_grid) - 1):
                log(f"  grid {iz + 1}/{len(z_grid)}  ({time.monotonic() - started:.1f} s)", flush=True)
    elapsed = time.monotonic() - started

    if not (np.all(np.isfinite(flux)) and np.all(flux >= 0)):
        raise ValueError("grid contains non-finite or negative flux")
    expected = reference_flux_uJy(config.ref_mag)
    if not np.allclose(flux[..., bands.index(config.ref_band)], expected):
        raise ValueError("reference-band normalization failed")

    metadata = {
        "generator": "make_grid.py",
        "version": __version__,
        "format_version": GRID_FORMAT_VERSION,
        "config": config.to_dict(),
        "shape": list(flux.shape),
        "n_nodes": int(np.prod(flux.shape[:-1])),
        "flux_unit": "uJy; normalized so that ref_filter has ref_mag AB",
        "reference_flux_uJy": float(expected),
        "wavelength_grid": {
            "n": int(assets.wave.size),
            "min_A": float(assets.wave.min()),
            "max_A": float(assets.wave.max()),
            "sha256": hashlib.sha256(np.ascontiguousarray(assets.wave).tobytes()).hexdigest(),
            "construction": (
                f"unique(geomspace(lam_lo, lam_hi, {config.n_wave}) "
                f"+ template nodes + {list(FEATURE_WAVELENGTHS)})"
            ),
        },
        "inputs": {
            "template_param": str(assets.templates.param_file),
            "templates": {
                path.name: {"path": str(path), "sha256": sha256_file(path)}
                for path in assets.templates.paths
            },
            "filter_manifest": str(assets.manifest_file),
            "filters": {
                band: {
                    "profile": row["profile"],
                    "sha256": row.get("sha256"),
                    "source_url": row.get("source_url"),
                }
                for band, row in assets.filter_rows.items()
            },
        },
        "data_provenance": {
            "templates": "EAZY (eazy-py) FSPS QSF v3, tweak_fsps_QSF_12_v3",
            "filters": {
                "repository": assets.manifest.get("repository"),
                "commit": assets.manifest.get("commit"),
            },
        },
        "software": _software_versions(),
        "flux_statistics": {
            "min": float(flux.min()),
            "max": float(flux.max()),
        },
        "elapsed_seconds": round(elapsed, 3),
    }
    log(f"Grid complete: {flux.shape}, {np.prod(flux.shape[:-1]):,} nodes, {elapsed:.1f} s")
    return TemplateGrid(
        flux=flux,
        redshift=z_grid,
        ebv=ebv_grid,
        template_names=list(assets.templates.names),
        filter_names=bands,
        ref_filter=config.ref_band,
        ref_mag=float(config.ref_mag),
        config_hash=assets.config_hash,
        metadata=metadata,
    )


def build_grid(
    config: GridConfig | None = None,
    *,
    base_dir: str | Path | None = None,
    template_param: str | Path | None = None,
    filter_dir: str | Path | None = None,
    manifest_file: str | Path | None = None,
    verify_filters: bool = True,
    threads: int = 2,
    progress_every: int = 50,
    log=print,
) -> TemplateGrid:
    """Load the inputs and build the grid (nothing is written). Defaults give 401x50x12x11."""
    config = config or GridConfig()
    assets = load_assets(
        config,
        base_dir=base_dir,
        template_param=template_param,
        filter_dir=filter_dir,
        manifest_file=manifest_file,
        verify_filters=verify_filters,
    )
    return _synthesize(assets, threads=threads, progress_every=progress_every, log=log)


# --------------------------------------------------------------------------------------
# Output and read-back
# --------------------------------------------------------------------------------------


def plot_diagnostics(
    grid: TemplateGrid, path: str | Path, templates: TemplateSet | None = None
) -> Path:
    """Plot band flux versus redshift (left) and the template rest-frame spectra (right)."""
    import matplotlib

    if not matplotlib.is_interactive():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = list(grid.filter_names)
    ref_index = bands.index(grid.ref_filter)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, band in enumerate(bands):
        axes[0].plot(
            grid.redshift,
            grid.flux[:, 0, 0, index] / grid.flux[:, 0, 0, ref_index],
            label=band,
        )
    axes[0].set(
        xlabel="Redshift",
        ylabel=f"Flux / {grid.ref_filter} flux",
        yscale="log",
        ylim=(1e-7, 30),
        title=f"Template 001, E(B-V)=0 (norm. {grid.ref_filter}={grid.ref_mag:g} AB)",
    )
    axes[0].legend(ncol=2, fontsize=8)
    if templates is not None:
        for index, (wave, sed) in enumerate(zip(templates.wave, templates.sed)):
            axes[1].plot(wave, sed / np.interp(5500.0, wave, sed), lw=0.8, label=f"{index + 1:02d}")
        axes[1].set(
            xlabel="Rest wavelength (Angstrom)",
            ylabel=r"Relative $f_\lambda$",
            xscale="log",
            yscale="log",
            xlim=(500, 60000),
            ylim=(1e-6, 1e3),
            title=f"{len(templates)} templates",
        )
        axes[1].legend(ncol=4, fontsize=8)
    else:
        axes[1].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_grid(
    grid: TemplateGrid,
    output_dir: str | Path = PACKAGE_DIR / DEFAULT_OUTPUT_DIRNAME,
    *,
    diagnostics: bool = True,
    templates: TemplateSet | None = None,
    write_npz: bool = True,
    log=print,
) -> dict[str, Path]:
    """Write the grid as npz + metadata json + (optionally) the diagnostics figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"fsps_{len(grid.filter_names)}band_grid.npz"
    metadata_path = output_dir / "grid_metadata.json"
    if write_npz:
        np.savez_compressed(npz_path, **grid.npz_kwargs())
        log(f"Wrote: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    metadata_path.write_text(
        json.dumps(grid.metadata, indent=2, sort_keys=True, ensure_ascii=False)
    )
    paths = {"grid": npz_path, "metadata": metadata_path}
    log(f"Wrote: {metadata_path}")
    if diagnostics:
        paths["diagnostics"] = plot_diagnostics(
            grid, output_dir / "grid_diagnostics.png", templates=templates
        )
        log(f"Wrote: {paths['diagnostics']}")
    return paths


def load_grid(path: str | Path) -> TemplateGrid:
    """Read back the npz written by save_grid."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return TemplateGrid(
            flux=np.asarray(data["flux"]),
            redshift=np.asarray(data["redshift"]),
            ebv=np.asarray(data["ebv"]),
            template_names=[str(name) for name in data["template_names"]],
            filter_names=[str(name) for name in data["filter_names"]],
            ref_filter=str(data["ref_filter"]),
            ref_mag=float(data["ref_mag"]),
            config_hash=str(data["config_hash"]),
            metadata=json.loads(str(data["metadata"])),
        )


@dataclass
class RunResult:
    """Return value of run(): grid, written paths, cache hit, total elapsed time."""

    grid: TemplateGrid
    paths: dict[str, Path]
    cached: bool
    elapsed: float


def run(
    config: GridConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
    template_param: str | Path | None = None,
    filter_dir: str | Path | None = None,
    manifest_file: str | Path | None = None,
    verify_filters: bool = True,
    threads: int = 2,
    progress_every: int = 50,
    force: bool = False,
    diagnostics: bool = True,
    log=print,
) -> RunResult:
    """Full pipeline with a content-hash cache: load -> reuse the cache on a hit -> else compute and write."""
    config = config or GridConfig()
    started = time.monotonic()
    base = Path(base_dir).expanduser().resolve() if base_dir else PACKAGE_DIR
    out = Path(output_dir) if output_dir else base / DEFAULT_OUTPUT_DIRNAME
    assets = load_assets(
        config,
        base_dir=base,
        template_param=template_param,
        filter_dir=filter_dir,
        manifest_file=manifest_file,
        verify_filters=verify_filters,
    )
    npz_path = out / f"fsps_{len(config.bands)}band_grid.npz"
    if not force and npz_path.is_file():
        try:
            cached_grid = load_grid(npz_path)
        except (EOFError, OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            # A truncated or otherwise unreadable cache (e.g. an interrupted write) must
            # never abort the run: warn and recompute instead.
            log(f"warning: cached grid {npz_path} is unreadable ({exc}); recomputing")
        else:
            if cached_grid.config_hash == assets.config_hash:
                log(f"content hash matches, reusing cache {npz_path} (use --force to recompute)")
                paths = save_grid(
                    cached_grid,
                    out,
                    diagnostics=diagnostics,
                    templates=assets.templates,
                    write_npz=False,
                    log=log,
                )
                return RunResult(
                    grid=cached_grid, paths=paths, cached=True, elapsed=time.monotonic() - started
                )

    log(
        f"Building grid: {config.n_z}x{config.n_ebv}x{len(assets.templates)}x{len(config.bands)}, "
        f"wavelength grid {assets.wave.size} points, config_hash={assets.config_hash[:12]}..."
    )
    grid = _synthesize(assets, threads=threads, progress_every=progress_every, log=log)
    paths = save_grid(grid, out, diagnostics=diagnostics, templates=assets.templates, log=log)
    return RunResult(grid=grid, paths=paths, cached=False, elapsed=time.monotonic() - started)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show an automatic "(default: ...)" only when it adds information to the help."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if "%(default)" in help_text or action.default is None:
            return help_text
        if action.nargs == 0 and isinstance(action.const, bool):
            # store_true / store_false: the flag itself already tells the default
            return help_text
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_grid.py",
        description="Build a (z, E(B-V), template, band) photometric flux grid from FSPS "
                    "templates, G23 dust attenuation and the Inoue14 IGM.",
        formatter_class=_HelpFormatter,
    )
    grid = parser.add_argument_group("Grid definition")
    grid.add_argument(
        "--bands",
        nargs="+",
        default=list(DEFAULT_BANDS),
        metavar="BAND",
        help="Bands to synthesize; the order sets the grid's last axis (must be in the manifest)",
    )
    grid.add_argument("--z-min", type=float, default=0.0, help="minimum redshift")
    grid.add_argument("--z-max", type=float, default=4.0, help="maximum redshift")
    grid.add_argument("--n-z", type=int, default=401, help="number of redshift grid points")
    grid.add_argument("--ebv-min", type=float, default=0.0, help="minimum E(B-V)")
    grid.add_argument("--ebv-max", type=float, default=1.0, help="maximum E(B-V)")
    grid.add_argument("--n-ebv", type=int, default=50, help="number of E(B-V) grid points")
    grid.add_argument("--rv", type=float, default=3.1, help="Rv of the G23 curve")
    grid.add_argument("--ref-band", default="UVISTA_Ks", help="reference band used for normalization")
    grid.add_argument("--ref-mag", type=float, default=25.0, help="AB magnitude of the reference band")
    grid.add_argument(
        "--n-wave",
        type=int,
        default=16000,
        help="dense log-spaced rest-frame wavelength points (native template nodes are added)",
    )

    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--base-dir", default=None, help="base directory for relative paths (default: directory of this script)")
    inputs.add_argument(
        "--template-param", default=None, help=f"template .param file (default {DEFAULT_TEMPLATE_PARAM})"
    )
    inputs.add_argument(
        "--filter-dir", default=None, help=f"filter root directory (default {DEFAULT_FILTER_DIRNAME})"
    )
    inputs.add_argument(
        "--manifest",
        default=None,
        help=f"filter manifest (default {DEFAULT_FILTER_DIRNAME}/{DEFAULT_MANIFEST_NAME})",
    )
    inputs.add_argument(
        "--no-verify",
        dest="verify_filters",
        action="store_false",
        help="skip the filter sha256 check (faster, but not recommended)",
    )

    output = parser.add_argument_group("Output and execution")
    output.add_argument(
        "--output-dir", default=None, help=f"output directory (default {DEFAULT_OUTPUT_DIRNAME})"
    )
    output.add_argument("--threads", type=int, default=2, help="cap on BLAS/OpenMP threads")
    output.add_argument(
        "--progress-every", type=int, default=50, help="print progress every N redshift steps (0 = silent)"
    )
    output.add_argument("--force", action="store_true", help="ignore the content-hash cache and recompute")
    output.add_argument(
        "--no-diagnostics", dest="diagnostics", action="store_false", help="do not produce the diagnostics figure"
    )
    output.add_argument("--quiet", action="store_true", help="print no log output at all")
    output.add_argument("--version", action="version", version=f"make_grid {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print
    try:
        config = GridConfig(
            bands=tuple(args.bands),
            z_min=args.z_min,
            z_max=args.z_max,
            n_z=args.n_z,
            ebv_min=args.ebv_min,
            ebv_max=args.ebv_max,
            n_ebv=args.n_ebv,
            rv=args.rv,
            ref_band=args.ref_band,
            ref_mag=args.ref_mag,
            n_wave=args.n_wave,
        )
        result = run(
            config,
            output_dir=args.output_dir,
            base_dir=args.base_dir,
            template_param=args.template_param,
            filter_dir=args.filter_dir,
            manifest_file=args.manifest,
            verify_filters=args.verify_filters,
            threads=args.threads,
            progress_every=args.progress_every,
            force=args.force,
            diagnostics=args.diagnostics,
            log=log,
        )
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    log("")
    log(f"grid    : {result.grid.shape}  (z, E(B-V), template, band)")
    log(f"nodes   : {result.grid.n_nodes:,}; elapsed {result.elapsed:.1f} s; cache hit: {result.cached}")
    log(f"bands   : {', '.join(result.grid.filter_names)}")
    log(f"template: {len(result.grid.template_names)} -> "
        f"{result.grid.template_names[0]} ... {result.grid.template_names[-1]}")
    log(f"norm    : {result.grid.ref_filter} = {result.grid.ref_mag:g} AB -> "
        f"{reference_flux_uJy(result.grid.ref_mag):.6f} uJy")
    log(f"hash    : {result.grid.config_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
