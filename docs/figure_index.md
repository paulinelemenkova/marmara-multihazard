# Figure index — script → output → manuscript figure

Each script writes a `Figure_NN.png` next to itself (300 dpi). The internal
output number is offset from the manuscript number because the manuscript adds
a study-area map (Fig. 1) and two workflow diagrams that are not scripted here.
Use this table to map each script to the figure as it appears in the paper.

| Script                          | Writes         | Manuscript figure | Content                                  |
|---------------------------------|----------------|-------------------|------------------------------------------|
| `Figure_03_framework.py`        | `Figure_03.png`| Fig. 2            | Conceptual multi-hazard framework        |
| `plot_geology_marmara.py`       | `Figure_04.png`| Fig. 3            | Geology & active tectonics               |
| `plot_landcover_marmara.py`     | `Figure_05.png`| Fig. 4            | Land use / land cover (CLC+)             |
| `plot_population_marmara.py`    | `Figure_06.png`| Fig. 5            | Population distribution                  |
| `plot_transport_overpass.py`    | `Figure_07.png`| Fig. 6            | Transport network (OpenStreetMap)        |
| `plot_windhazard_marmara.py`    | `Figure_08.png`| Fig. 7            | Wind hazard (mean speed at 100 m)        |
| `plot_xai_schematic.py`         | `Figure_10.png`| Fig. 9            | Explainable-AI (SHAP) schematic          |
| `plot_seismic_marmara.py`       | `Figure_11.png`| Fig. 10           | Seismicity & active tectonics            |
| `plot_tsunami_marmara.py`       | `Figure_12.png`| Fig. 11           | Tsunami exposure                         |
| `plot_windrisk_marmara.py`      | `Figure_13.png`| Fig. 12           | Wind risk (hazard × urban exposure)      |
| `plot_riskheatmap_marmara.py`   | `Figure_15.png`| Fig. 14           | Multi-hazard risk-factor heatmap         |
| `compress_figures.py`           | —              | —                 | Utility: caps figure PNG file sizes      |

Not scripted here (produced with other tools): Fig. 1 (study-area NUTS-1 map),
Fig. 8 (ML workflow diagram), Fig. 13a–d (composite risk-panel maps).
