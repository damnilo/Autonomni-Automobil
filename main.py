"""
MetaDrive Keyboard Control Script
===================================
Control a vehicle in MetaDrive simulator using keyboard inputs with
progressive ramping, asymmetric release behavior, and persistent logging.

Installation:
    pip install metadrive-simulator pygame

Controls:
    A / D     — Steer left / right  (slow auto-center on release, decay ~0.03/frame)
    W         — Accelerate          (slow auto-decay on release, ~0.02/frame)
    S         — Brake / throttle cut (instant throttle → 0, stays 0 on release)
    SPACE     — Reset steering & throttle to zero instantly
    Q / ESC   — Quit and save log

Tunable parameters (see CONFIG in game.py):
    STEER_STEP      = 0.05   increment per frame while key held
    STEER_DECAY     = 0.07   subtracted per frame on release (toward 0)
    THROTTLE_STEP   = 0.05   increment per frame while W held
    THROTTLE_DECAY  = 0.02   subtracted per frame on W release
    BRAKE_VALUE     = -0.3   applied while S is held (set to 0.0 to just cut throttle)
"""

from datetime import datetime
import os

from utils.logger import ActionLogger
from manual_control.game import Game

from utils.env_randomizer import get_random_metadrive_config
from manual_control.game import CONFIG
from collect_dataset import run_dataset_collection
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == '__main__' :
    run_dataset_collection()
