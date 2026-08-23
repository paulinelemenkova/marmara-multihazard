# Data

The scripts read source datasets from a single root directory, `DATA_ROOT`.
By default this resolves to this `data/` folder; override it to point at your
own local copy:

```bash
export DATA_ROOT=/path/to/your/DATA      # macOS/Linux
# setx DATA_ROOT "D:\path\to\DATA"       # Windows (new shell after)
```

## Bundled in this repository
| File                       | Description                                   | License |
|----------------------------|-----------------------------------------------|---------|
| `tuik_pop_2022.csv`        | Province population (2022)                     | TÜİK, open |
| `overpass_transport.json`  | Cached OpenStreetMap/Overpass transport query | ODbL, © OpenStreetMap contributors |

## Not bundled — download from the open providers below
These are large and/or separately licensed. Place them under `DATA_ROOT` using
the subfolder names the scripts expect (see each script's header comment).

| Dataset                         | Provider / link                                  |
|---------------------------------|--------------------------------------------------|
| Land cover (CLC+ Backbone 2018) | Copernicus LMS — https://land.copernicus.eu      |
| Bathymetry & relief (GEBCO)     | GEBCO — https://www.gebco.net                    |
| Province boundaries (ADM1)      | geoBoundaries — https://www.geoboundaries.org    |
| Geology & active faults (500k)  | MTA Türkiye — https://www.mta.gov.tr             |
| Province population (ADNKS)     | TÜİK — https://data.tuik.gov.tr                  |
| Sentinel-1/-2 imagery           | ESA Copernicus — https://dataspace.copernicus.eu |
| Copernicus DEM (GLO-30)         | ESA/Copernicus — https://spacedata.copernicus.eu |
| ERA5 reanalysis                 | ECMWF CDS — https://cds.climate.copernicus.eu    |
| OpenStreetMap (roads/rail)      | OSM — https://www.openstreetmap.org (ODbL)       |
| Strong-motion records           | AFAD TADAS — https://tadas.afad.gov.tr           |
| Earthquake catalogue            | USGS ANSS ComCat — https://earthquake.usgs.gov   |
| Global Wind Atlas (100 m)       | DTU/World Bank — https://globalwindatlas.info    |

The Istanbul Metropolitan Municipality building inventory is available on
request from the provider and is not redistributed here.
