import configparser
import os
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
file_path = os.path.join(ROOT_DIR,"config/config.properties")

config = configparser.ConfigParser()
config.read(file_path)
