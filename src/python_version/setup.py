import subprocess
import sys
from importlib.metadata import distribution, PackageNotFoundError

REQUIRED_PACKAGES = [
    'numpy',
    'matplotlib',
]

for package in REQUIRED_PACKAGES:
    try:
        dist = distribution(package)
        print('{} ({}) is installed'.format(dist.metadata['Name'], dist.version))
    except PackageNotFoundError:
        print('{} is NOT installed'.format(package))
        subprocess.call([sys.executable, "-m", "pip", "install", package])