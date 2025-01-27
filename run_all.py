import os
import json
import argparse
import subprocess
import signal
import sys

from tqdm import tqdm

def signal_handler(sig, frame):
    print('Execution interrupted. Exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all the scripts in the given directory')
    parser.add_argument('directory', help='The directory where the config scripts are located')
    args = parser.parse_args()

    directory = args.directory
    loop = tqdm(os.listdir(directory), desc="\033[91mRunning...\033[0m")
    for config in loop:
        with open(os.path.join(directory, config)) as f:
            data = json.load(f)
            outdir = os.path.join('data', data['name'])
        loop.set_description(f"\033[91mRunning {config}\033[0m")
        print(" ")
        try:
            subprocess.call([
                'python', 'main.py', os.path.join(directory, config),
            ])
            print(" ")
            subprocess.call([
                'python', 'analyse.py', outdir,
            ])
        except:
            continue