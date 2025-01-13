Preparing Forecast Data
=========================

RLHAB already comes with examples ERA5 forecasts and synthetic forecasts to get started.
This page explains  how to download/generate forecasts for other regions and/or time periods.


Downloading ERA5 forecasts
___________________________________________

Download ERA5 Reanalysis Forecasts from ECWMF data store. You will need to make an account first
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview


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

.. note:: ERA5 forecasts may need to be downloaded in batches depending on the region or time periods
    of interest. The Copernicus server does not support downloads over 10 gigs.

Generating Synthetic Forecasts
___________________________________________

See Synth winds [ToDO]
