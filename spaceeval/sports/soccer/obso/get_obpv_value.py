from ...c_obso_repo import Metrica_IO as mio
from ...common import cmath

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_obpv(Metrica_df, tracking_home, tracking_away):
    return pd.DataFrame()


def _calculate_obpv(ppcf, transition, pitch_weight, tracking, attack_direction=0):
    """
    Calculate OBPV value in a single frame

    Parameters
    ----------
    ppcf : np.ndarray
        Pitch control field (PPCF) for the frame
    transition : np.ndarray
        Transition matrix for the frame
    pitch_weight : np.ndarray
        Pitch weight matrix
    tracking : pd.DataFrame
        Tracking data for the frame
    attack_direction : int
        Attack direction, 1 for left to right, -1 for right to left
    
    Returns
    -------
    obpv : np.ndarray
        Offensive Ball Possession Value (OBPV) for the frame
    transition : np.ndarray
        Transition matrix adjusted for ball position

    """
    transition = np.array(transition)

    try:
        ball_grid_x = int((tracking["ball_x"] + 52.5) // (105 / 50))
    except ValueError:
        ball_grid_x = 0

    try:
        ball_grid_y = int((tracking["ball_y"] + 34) // (68/32))
    except ValueError:
        ball_grid_y = 0

    ball_grid_x = cmath.clamp(ball_grid_x, 0, 49)
    ball_grid_y = cmath.clamp(ball_grid_y, 0, 31)

    transition = transition[
        31 - ball_grid_y: 63 - ball_grid_y,
        49 - ball_grid_x: 99 - ball_grid_x
    ]

    if attack_direction < 0:
        pitch_weight = np.fliplr(pitch_weight)
    elif attack_direction > 0:
        pass
    else:
        print("Input attack direction is invalid. Should be 1 or -1")

    obpv = ppcf * transition * pitch_weight
    return obpv, transition
