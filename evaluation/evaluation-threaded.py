import subprocess
import env.config.env_config as env_config
from env.config.env_config import env_params

def update_env_params(key, value):
    # Dynamically update the global env_params in the module
    globals()["env_params"][key] = value

# List of months (or arguments for each process)
months_H1 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
months_H2 = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#To kill all screens
#screen -ls | grep evaluate_ | awk '{print $1}' | xargs -I {} screen -S {} -X quit

# Command template
script = "evaluation/evaluate_DUAL.py"

for month in months_H1:
    #print(month)
    env_config.set_param("eval_month",month)
    env_config.set_param("era_netcdf", "ERA5-H1-2023-USA.nc")
    env_config.set_param("synth_netcdf", f"SYNTH-{month}-2023-USA-UPDATED.nc")
    #env_params["eval_month"] = month
    #env_params["era_netcdf"] = "ERA5-H1-2023-USA.nc"
    #env_params["synth_netcdf"] = f"SYNTH-{month}-2023-USA-UPDATED.nc"

    print(env_params["synth_netcdf"])


    # Create a unique screen session name for each month
    session_name = f"evaluate_{month.lower()}"

    # Construct the command to run the script with the month as an argument
    command = f"screen -dmS {session_name} python {script} --month {month}"

    print(command)

    # Launch the screen session
    subprocess.run(command, shell=True)

for month in months_H2:

    env_params["eval_month"] = month
    env_params["era_netcdf"] = "ERA5-H2-2023-USA.nc"
    env_params["synth_netcdf"] = "SYNTH-" + month + "-2023-USA-UPDATED.nc"


    # Create a unique screen session name for each month
    session_name = f"evaluate_{month.lower()}"

    # Construct the command to run the script with the month as an argument
    command = f"screen -dmS {session_name} python {script} --month {month}"

    # Launch the screen session
    subprocess.run(command, shell=True)

print("All processes have been launched in separate screens.")
