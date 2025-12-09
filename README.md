# DRP Processing

Python toolkit for working with directional reflectance profile (DRP) image stacks. It handles data prep (RGB → grayscale, blurred background generation), fast DRP stack construction with on-disk caching, per-pixel orientation/moment analysis, and simple visualisation utilities.

## Repository layout
- `main.py` – minimal example that loads an image stack, plots the mean DRP, computes direction maps, and does a quick peak finding pass.
- `src/` – core library code:
  - `imagepack.py` – `ImagePack` loader with caching, background subtraction, angle slicing, DRP extraction, and plotting helpers.
  - `drp_compute.py`, `drp_direction.py`, `drp_plot.py` – DRP stack creation, spherical moment/direction maps, and plotting utilities.
  - `line_detection.py` – Hough-based line/angle detection helpers.
  - `config.py`, `image_io.py`, `paths.py` – configuration, I/O, and data path helpers.
  - `roi.py`, `imageparam.py` – lightweight ROI and parameter helpers kept for compatibility.
- `scripts/` – standalone preprocessing scripts:
  - `rgb2grey.py` converts `data/raw` RGB images to grayscale into `data/processed`.
  - `bg_blur.py` builds a heavily blurred background set in `data/background` for subtraction.
- `data/` – expected data root; contains `raw/`, `processed/`, `background/`, and `cache/` (memmap + metadata written automatically).
- `exp_param.yaml` – DRP acquisition parameters for the dataset (phi/theta ranges and counts).
- `main.ipynb`, `map_visualise.ipynb` – interactive exploration notebooks.

## Quick start
1) Install dependencies (Python 3.11+ recommended):
```bash
python -m venv .venv
.venv\Scripts\activate   # PowerShell on Windows
pip install -r requirements.txt
```

2) Prepare data:
- Place raw RGB captures in `data/raw` (filenames will be reused).
- Convert to grayscale: `python scripts/rgb2grey.py` (writes to `data/processed`).
- Generate blurred backgrounds (optional but enables subtraction): `python scripts/bg_blur.py` (writes to `data/background`).

3) Configure angles in `exp_param.yaml`:
```yaml
data_serial: 7        # bump this to invalidate cached DRP when data changes
th_min: 10            # theta start (deg)
th_max: 65            # theta end   (deg)
th_num: 12            # theta samples
ph_min: 0             # phi start (deg)
ph_max: 351           # phi end   (deg)
ph_num: 40            # phi samples
```
`data_serial` is stored alongside the cached memmap; changing it forces a rebuild.

4) Run the example pipeline:
```bash
python main.py
```
This loads grayscale images from `data/processed`, subtracts blurred backgrounds if present, builds (or reuses) a cached DRP stack in `data/cache/drp.dat`, plots the mean DRP, computes a direction map + mask, and shows a simple column-intensity peak overlay.

## Core usage (library)
```python
from src import ImagePack
from src.drp_direction import drp_direction_map, drp_mask_angle

# Load images with optional angle down-sampling and background subtraction
imp = ImagePack(
    data_root="data",
    angle_slice=(2, 2),            # take every 2nd phi/theta to cut resolution
    img_format="jpg",
    use_cached_stack=True,         # reuse data/cache/drp.dat when it matches config
    subtract_background=True,      # subtracts images in data/background if present
)

# Mean DRP over the image or from a single pixel
mean_drp = imp.get_mean_drp()      # shape [phi, theta]
pixel_drp = imp.drp((100, 200))    # per-pixel DRP from the cached stack

# Per-pixel DRP direction and a magnitude-weighted mask near 90° ± 45°
mag_map, deg_map = drp_direction_map(imp, display=False)
mask = drp_mask_angle(mag_map, deg_map, orientation=90, threshold=45)

# Apply a mask to images (e.g., to focus on an ROI) and rebuild metrics
imp.mask_images(mask, normalize=True)
```

### Additional helpers
- `src.line_detection.hough_transform` / `find_hough_peaks` / `rotate_image_to_orientation` for line detection and alignment.
- `src.drp_direction.spherical_descriptor` and `spherical_descriptor_maps` for per-pixel 3D orientation descriptors, anisotropy scores, and dominant plane angles.
- `ImagePack.plot_drp` (wraps `drp_plot.plot_drp`) to visualise any DRP array in stereo or direct projection.

## Data & caching notes
- Data is addressed via `data_root` (defaults to `data/` relative to the repo) with expected subfolders `raw/`, `processed/`, `background/`, and `cache/`.
- The DRP stack is stored as a NumPy memmap at `data/cache/drp.dat` with metadata in `data/cache/data_config.yaml`. The cache is automatically rebuilt when:
  - the angle slice passed to `ImagePack` differs from the cached slice, or
  - `data_serial` in `exp_param.yaml` changes, or
  - the on-disk shape mismatches the expected phi/theta counts.
- `angle_slice` lets you down-sample the phi/theta grid (e.g., `(2, 2)` halves both resolutions) before stack construction for faster experimentation.

## Notebooks
- `main.ipynb` follows the scripted pipeline with richer plots.
- `map_visualise.ipynb` contains interactive DRP/angle visualisations. Activate the virtual environment first so notebooks use the same dependencies.

## Troubleshooting
- No images found: ensure grayscale images exist in `data/processed` (correct extension set by `img_format`).
- Background subtraction errors: background counts and dimensions must match the processed set; regenerate via `scripts/bg_blur.py`.
- Cache mismatch errors: bump `data_serial` or delete `data/cache` to force a clean rebuild.
