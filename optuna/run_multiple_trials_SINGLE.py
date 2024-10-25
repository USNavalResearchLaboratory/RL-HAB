"""
Example of how to run multiple processes of optuna-multi-SINGLE.py for faster training
"""

import subprocess
import psutil
import time
import optuna_config

# Number of processes to run in parallel
num_processes = optuna_config.n_threads

# List to hold the processes
processes = []

for _ in range(num_processes):
    process = subprocess.Popen(['python3', 'optuna/optuna-multi-SINGLE.py'])
    processes.append(process)

try:
    while True:
        time.sleep(1)  # Run indefinitely, monitoring the processes
except KeyboardInterrupt:
    print("Received KeyboardInterrupt, terminating processes...")

    for process in processes:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):  # Terminate child processes
            child.terminate()
        parent.terminate()  # Terminate parent process

    for process in processes:
        process.wait()  # Ensure all processes are terminated