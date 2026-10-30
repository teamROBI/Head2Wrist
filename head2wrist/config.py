"""Environment configuration shared by all command-line tools.

Set these variables before starting Python. Explicit command-line paths override defaults.
"""
import os
import sys
from pathlib import Path


def _path(name, default):
    return str(Path(os.environ.get(name, default)).expanduser().resolve())


H2W_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
H2W_DATA_ROOT = _path('H2W_DATA_ROOT', Path(H2W_REPO_ROOT) / 'data')
H2W_OUTPUT_ROOT = _path('H2W_OUTPUT_ROOT', Path(H2W_DATA_ROOT) / 'outputs')
H2W_EGODEX_ROOT = _path('H2W_EGODEX_ROOT', Path(H2W_DATA_ROOT) / 'egodex' / 'test')
H2W_URDF_ROOT = _path('H2W_URDF_ROOT', Path(H2W_REPO_ROOT) / 'third_party' / 'ai_worker' / 'ffw_description')
H2W_ROBOTIS_ROOT = _path('H2W_ROBOTIS_ROOT', Path(H2W_REPO_ROOT) / 'third_party' / 'robotis_lab')
H2W_SAM3D_ROOT = _path('H2W_SAM3D_ROOT', Path(H2W_REPO_ROOT) / 'third_party' / 'sam-3d-objects')
H2W_PYTHON = os.environ.get('H2W_PYTHON', sys.executable)
H2W_LEROBOT_PYTHON = os.environ.get('H2W_LEROBOT_PYTHON', str(Path(H2W_REPO_ROOT) / 'third_party' / 'lerobot' / '.venv' / 'bin' / 'python'))
H2W_SAM3D_PYTHON = os.environ.get('H2W_SAM3D_PYTHON', str(Path(H2W_SAM3D_ROOT) / '.venv' / 'bin' / 'python'))

__all__ = [name for name in globals() if name.startswith('H2W_')]
