"""
Lorenz Attractor Professional Simulation Suite

A comprehensive toolkit for simulating, visualizing, and analyzing the Lorenz attractor
and related chaotic dynamical systems.
"""

__version__ = "2.0.0"
__author__ = "Jose Solis Rosales"
__email__ = "josesolisrosales@linux.com"

from .core.lorenz import LorenzSystem
from .core.simulator import Simulator
from .visualization.plotter import LorenzPlotter
from .visualization.realtime import RealtimeVisualizer
from .export.video import VideoExporter
from .export.data import DataExporter

__all__ = [
    "LorenzSystem",
    "Simulator", 
    "LorenzPlotter",
    "RealtimeVisualizer",
    "VideoExporter",
    "DataExporter",
]