#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:52:19 2020

Module for measuring player velocities, smoothed using a Savitzky-Golay filter, with Metrica tracking data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)

"""
import numpy as np
import pandas as pd
import scipy.signal as signal


def _smooth_velocity(values, window, polyorder):
    """Preserve missing windows without fitting NaNs at the signal edges."""
    if len(values) < window:
        raise ValueError("Velocity series must be at least as long as the smoothing window.")
    # Interior convolution already propagates NaNs. Recent SciPy versions
    # reject NaNs in the default edge polynomial fit instead of returning NaNs.
    result = signal.savgol_filter(values, window, polyorder, mode="constant")
    half = window // 2
    if half:
        for edge, output, fitted in (
            (slice(0, window), slice(0, half), slice(0, half)),
            (slice(-window, None), slice(-half, None), slice(-half, None)),
        ):
            segment = np.asarray(values)[edge]
            if np.isfinite(segment).all():
                result[output] = signal.savgol_filter(segment, window, polyorder)[fitted]
            else:
                result[output] = np.nan
    return result


def calc_player_velocities(
    team: pd.DataFrame,
    smoothing: bool = True,
    filter_: str = "Savitzky-Golay",
    window: int = 5,
    polyorder: int = 0,
    maxspeed: float = 12,
) -> pd.DataFrame:
    """calc_player_velocities( tracking_data )

    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data

    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN.

    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)

    # Get the player ids
    player_ids = np.unique([c[:-2] for c in team.columns if c[:4] in ["Home", "Away"]])

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team["Time [s]"].diff()

    # estimate velocities for players in team
    for player in player_ids:  # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player + "_x"].diff() / dt
        vy = team[player + "_y"].diff() / dt

        if maxspeed > 0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt(vx**2 + vy**2)
            vx[raw_speed > maxspeed] = np.nan
            vy[raw_speed > maxspeed] = np.nan

        if smoothing:
            if filter_ == "Savitzky-Golay":
                # calculate velocity
                vx = _smooth_velocity(vx, window, polyorder)
                vy = _smooth_velocity(vy, window, polyorder)
            elif filter_ == "moving average":
                ma_window = np.ones(window) / window
                # calculate velocity
                vx = np.convolve(vx, ma_window, mode="same")
                vy = np.convolve(vy, ma_window, mode="same")

        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt(vx**2 + vy**2)

    return team


def remove_player_velocities(team: pd.DataFrame) -> pd.DataFrame:
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [
        c
        for c in team.columns
        if c.split("_")[-1] in ["vx", "vy", "ax", "ay", "speed", "acceleration"]
    ]  # Get the player ids
    team = team.drop(columns=columns)
    return team
