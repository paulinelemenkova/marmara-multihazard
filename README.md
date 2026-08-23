# Multi-Hazard Risk Assessment of the Marmara Basin — Analysis & Figure Code

Reproducible Python code accompanying the manuscript *"Multi-Hazard Risk
Assessment of the Marmara Basin: A Deep Learning Framework for Istanbul"*
(currently under peer review). It generates the study's geospatial figures —
geology, land cover, population, transport, seismicity, tsunami exposure, wind
hazard/risk, the multi-hazard risk heatmap, and the framework/XAI schematics —
from open geospatial datasets.

## Repository layout
```
marmara-multihazard/
├── scripts/          # figure-generation & analysis scripts (one per figure)
├── data/             # small bundled inputs + where to get the large ones
│   └── README.md
├── figures/          # output PNGs are written here (git-ignored)
├── docs/
│   └── figure_index.md   # script → output → manuscript-figure mapping
├── requirements.txt  # pip environment
├── environment.yml   # conda environment
├── CITATION.cff      # how to cite this software
├── .zenodo.json      # metadata for the Zenodo archive
└── LICENSE           # MIT (code)
```

## Quick start
```bash
# 1. Environment (choose one)
conda env create -f environment.yml && conda activate marmara-multihazard
# or:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Point DATA_ROOT at your datasets (defaults to ./data)
export DATA_ROOT=/path/to/your/DATA

# 3. Run any figure script (writes Figure_NN.png next to the script)
python scripts/plot_seismic_marmara.py
python scripts/plot_tsunami_marmara.py
```

Several scripts accept `--inspect` (print dataset columns) or `--selftest`
(check the plotting pipeline without data) — see each script's header.

See **`docs/figure_index.md`** for which script produces which manuscript figure,
and **`data/README.md`** for dataset sources and licenses.

## Data & licensing
- **Code:** MIT (see `LICENSE`).
- **Bundled OSM snapshot** (`data/overpass_transport.json`): © OpenStreetMap
  contributors, ODbL.
- **External datasets** are redistributed by their original providers under
  their own terms; this repository only points to them (`data/README.md`).

## Citing
If you use this code, please cite the software via `CITATION.cff` (and the
associated article once published). A archived release with a DOI is available
on Zenodo — see below.

## Archived release
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

*(Replace `XXXXXXX` with your Zenodo record ID after the first release.)*
