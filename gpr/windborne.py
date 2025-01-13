import numpy as np
import requests
import re
import os


def download_forecast(initialization_time):
    """
    Downloads a geopotential forecast from the Windborne API
    This will create a file for each hour in the forecast at 6 hour resolution, and return those filenames
    These files will contain the 500mb geopotential forecasts for the entire globe in npy format
    initialization_time is expected to be a ISO string
    """
    url = f"https://forecasts.windbornesystems.com/api/v1/gridded/historical/500/geopotential"
    files = []
    expected_file_size = 720*1440*4 + 128  # 720x1440 grid, 4 bytes per float, 128 bytes of numpy header

    for hour in range(24, 24*7 + 1, 24):
        response = requests.get(url, params={'forecast_hour': hour, 'initialization_time': initialization_time}, allow_redirects=True, stream=True)
        response.raise_for_status()

        # take the filename from the response header; you're welcome to change this
        # NOTE: you may want to change this download location
        filename = re.findall("filename=(.+)", response.headers['content-disposition'])[0]
        if os.path.exists(filename) and os.path.getsize(filename) == expected_file_size:
            print(f"{filename} already downloaded")
            files.append(filename)
            continue

        print(f"Downloading {filename}")

        with open(filename, "wb") as handle:
            for data in response.iter_content(chunk_size=1024*1024):
                handle.write(data)

        files.append(filename)

    return files


if __name__ == "__main__":
    initialization_time = "2024-03-12T12:00:00Z"
    forecast = download_forecast(initialization_time)
    print(f"Downloaded {len(forecast)} files")
    data = np.load(forecast[0])
    print("First forecast has shape", data.shape)