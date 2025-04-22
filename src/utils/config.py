import os
from pathlib import Path # don't know if needed 

# Get the project root (LINGO folder)
def get_project_root():
    # Assuming this file is in src/utils/config.py
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils folder
    src_dir = os.path.dirname(current_dir)  # src folder
    project_root = os.path.dirname(src_dir)  # LINGO folder
    # print(f"current_dir: {current_dir}")
    # print(f"src_dir: {src_dir}")
    # print(f"project_root: {project_root}")
    return project_root

# Define paths
PROJECT_ROOT = get_project_root()
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'text')
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')


print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")
print(f"Configs directory: {CONFIGS_DIR}")
#get_project_root()