"""Lets scripts run from a fresh clone without `pip install -e .`."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
