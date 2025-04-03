import subprocess
import env.config.env_config as env_config
from env.config.env_config import env_params

def update_env_params(key, value):
    # Dynamically update the global env_params in the module
    globals()["env_params"][key] = value

#env_params["render_mode"] = "None"

# List of months (or arguments for each process)
months_H1 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
months_H2 = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#To kill all screens
#screen -ls | grep evaluate_ | awk '{print $1}' | xargs -I {} screen -S {} -X quit

# Command template
script = "evaluation/evaluate_DUAL.py"

for month in months_H1:
    #print(month)
    eval_month = month
    era_netcdf = env_params['era_netcdf']
    synth_netcdf = f"../../../FLOW2D/forecasts/SYNTH-{month}-2023-SEA-UPDATED.nc"

    # Create a unique screen session name for each month
    session_name = f"evaluate_{month.lower()}"

    # Construct the command to run the script with the month as an argument
    command = f"screen -dmS {session_name} python {script} --month {month}"

    command = (
        f"screen -dmS {session_name} python {script} "
        f"--month {eval_month} "
        f"--era_netcdf {era_netcdf} "
        f"--synth_netcdf {synth_netcdf}"
    )

    print(command)

    # Launch the screen session
    subprocess.run(command, shell=True)

for month in months_H2:
    #print(month)
    eval_month = month
    era_netcdf = env_params['era_netcdf']
    synth_netcdf = f"../../../FLOW2D/forecasts/SYNTH-{month}-2023-SEA-UPDATED.nc"

    # Create a unique screen session name for each month
    session_name = f"evaluate_{month.lower()}"

    # Construct the command to run the script with the month as an argument
    command = f"screen -dmS {session_name} python {script} --month {month}"

    command = (
        f"screen -dmS {session_name} python {script} "
        f"--month {eval_month} "
        f"--era_netcdf {era_netcdf} "
        f"--synth_netcdf {synth_netcdf}"
    )

    print(command)

    # Launch the screen session
    subprocess.run(command, shell=True)

print("All processes have been launched in separate screens.")
