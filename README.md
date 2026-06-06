# mvtxm_pal

Python package for TXM (Transmission X-ray Microscopy) XANES data processing, developed for the PAL (Pohang Accelerator Laboratory) beamline.

## Features

- **XANES fitting** — polynomial fitting, white-line position extraction, calibration
- **Image registration** — StackReg, Phase Correlation (PC), MRTV multi-resolution registration
- **I/O tools** — load/save TIF stacks, HDF5, background subtraction, median filtering
- **Data management** — organized output directories, colormap generation, statistics
- **GUI tools** — interactive edge selector, RGB stack viewer, particle segmentation

## Installation

```bash
git clone https://github.com/Hwiho/mvtxm_pal.git
cd mvtxm_pal
pip install -e ..
```

### Dependencies

```bash
pip install -r requirements.txt
```

> `txm_sandbox` is an optional dependency used for MRTV registration (`registration.py`).

## Package Structure

```
mvtxm_pal/
├── __init__.py               # package entry point
├── fit_xanes.py              # XANES fitting and colormap generation
├── io_pal.py                 # TIF/HDF5 I/O, background subtraction
├── registration.py           # image registration (StackReg, PC, MRTV)
├── misc_pal.py               # data management and saving utilities
├── imgcrop.py                # interactive image cropping (ROI)
├── binning.py                # image binning
│
├── scripts/                  # end-to-end processing pipelines
│   ├── main_PAL_auto.py          # batch auto-processing (proj/bkg folders)
│   ├── main_PAL_stack_auto.py    # batch processing (mean stack + energy.txt)
│   ├── main_PAL_single.py        # single scan processing
│   ├── main_PAL_auto_h5_refit.py # re-fit from saved HDF5
│   └── h5_refit.py               # HDF5 re-fitting utility
│
├── gui/                      # GUI applications
│   ├── Basic_gui.py              # main post-processing GUI
│   ├── Stack_rgb.py              # RGB stack viewer and colormap overlay
│   └── particle_segmentation.py  # particle segmentation tool
│
└── util/                     # standalone analysis utilities
    ├── colormap_generator.py
    ├── particle_analysis.py
    ├── bulk_spectrum.py
    ├── statistics_analysis.py
    ├── erosion_analysis.py
    ├── thickness_registration.py
    └── ...
```

## Quick Start

### Batch processing (stack format)

```python
# Each target_folder contains:
#   mean_proj_stack.tif, mean_back_stack.tif, energy.txt
from mvtxm_pal.scripts import main_PAL_stack_auto

# Or run directly:
# python scripts/main_PAL_stack_auto.py
```

### XANES fitting

```python
from mvtxm_pal import fit_xanes, iotools_pal

txm = iotools_pal()
txm.load_processed_h5("data.h5")

fitting = fit_xanes(txm.img, txm.eng, peakref=[8368.2, 8370.6])
fitting.set_thickness()
fitting.threshold(0.03)
fitting.polynomial_second_fit_separate(fit_num=3, ev_step=1)
fitting.hsvcolormap(value=3)
```

## Author

Hwiho Kim — PAL-XFEL / PLS-II beamline user
