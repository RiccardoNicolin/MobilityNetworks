import os
import json
import argparse
import subprocess
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all the scripts in the given directory')
    parser.add_argument('directory', help='The directory where the config scripts are located')
    args = parser.parse_args()

    directory = args.directory
    loop = tqdm(os.listdir(directory), desc="\033[91mRunning...\033[0m")
    for config in loop:
        with open(os.path.join(directory, config)) as f:
            data = json.load(f)
            outdir =  os.path.join('data', data['name'])
        loop.set_description(f"\033[91mRunning {config}\033[0m")
        print(" ")
        subprocess.call([
            'python', 'main.py', os.path.join(directory, config),
            ])
        print(" ")
        subprocess.call(
            [
                'python' 'analyse.py',
                outdir,
            ]
        )