Preparing Forecast Data
=========================

RLHAB requires at least one 4D forecast to run (RLHAB_env_SINGLE) in the ERA5 Pre Sep 2024 netcdf4 format, which is not provided due to file size.  Other forecast data sources can work as long as they are processed into the correct format.  
`ERAS-Utils <https://github.com/tkschuler/ERA5-Utils/>`_ provides more guidance on this formatting.


Downloading ERA5 forecasts
___________________________________________

The easiest way to get started is to download an ERA5 Reanalysis Forecast on individual pressure levels from the `ECMWF CDS <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview/>`_. 
You will need to make an account first.


.. list-table:: Default Parameters for downloading ERA5 Reanalysis NETCDF4 File
   :widths: 25 50
   :header-rows: 1

   * - Product Type
     - Reanalysis
   * - Variables
     - Geopotential, U-component of wind, V-component of wind, Temperature (optional)
   * - Pressure Level
     - 20-400 hPa
   * - Data format
     - NetCDF4
   * - Time 
     - At least 12 hour resolution 

Equatorial regions year-round and mid-latitudes in the summer have the best wind diversity for HAB navigation. 

For more advanced users, higher vertical resolution (recommended) ERA5 on model levels can be downloaded and processed.  ERA5 data on model levels comes in hybrid-sigma coordinates, which are not compatible with 
RLHAB, and must be converted to pressure levels using the `ERAS-Utils <https://github.com/tkschuler/ERA5-Utils/>`_

.. note:: ERA5 forecasts may need to be downloaded in batches depending on the region or time periods
    of interest. The Copernicus server does not support downloads over 10 gigs.

Generating Synthetic Forecasts
___________________________________________
The SynthWinds notebook is designed to process radiosonde data and export synthetic wind forecasts in NetCDF format. 
It includes functionality for radiosonde data aggregation, smoothing, I/O operations, and visualization. 
Radiosonde sounding data needs to be downloaded seperately from University of Wyoming's `Upper Air Soundings Dataset <https://weather.uwyo.edu/upperair/sounding_legacy.html>`_. 
`AnnualWyomingDownload.py` from `Radiowinds <https://github.com/tkschuler/RadioWinds/>`_ provides the script `AnnualWyomingDownload.py` to batch download individual soundings for a region and year using
the python package `Siphon <https://unidata.github.io/siphon/latest/index.html>`_ and organize the downloads into the same structure SynthWinds requires: 


 ::

    soundings_data
    ├── ABQ - 72365
    │   ├── 2022
    │   └── 2023
    │       └── 1
    │           ├──ABQ-2023-1-1-0.csv
    │           ├── ABQ-2023-1-1-12.csv
    │           ├── ABQ-2023-1-2-00.csv
    │           └── ...
    │       ├── 2
    │       └── ...
    ├── SLC - 72572
    └── ...

Steps for generating synthetic winds forecasts:

  1. Download the radiosonde data using `AnnualWyomingDownload.py` from RadioWinds

  2. Aggregate Radiosonde Data

  3. Filters radiosonde data for desired altitude, lat, and long ranges

  4. Interpolate missing data

  5. Post-Processing

  6. Visualization (optional)

  7. Reshape Data for NetCDF Export


Synthwinds is currently set up to download and process radiosonde data for the SouthWestern United States as an example. 

Generating SynthWinds Quickstart
___________________________________________

Clone RadioWinds, and run `AnnualWyomingDownload.py` with `config.py` as is.  This will download all radiosondes in North America for 2023. 