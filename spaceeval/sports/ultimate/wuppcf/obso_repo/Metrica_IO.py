#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:18:49 2020

Module for reading in Metrica sample data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)
"""

import csv as csv
from typing import Tuple

import pandas as pd


def read_event_data(event_file: str) -> pd.DataFrame:
    """
    read_event_data(DATADIR,game_id):
    read Metrica event data  for game_id and return as a DataFrame
    """
    events = pd.read_csv(event_file)
    return events


def tracking_data(tracking_file: str, teamname: str) -> pd.DataFrame:
    """
    tracking_data(DATADIR,game_id,teamname):
    read Metrica tracking data for game_id and return as a DataFrame.
    teamname is the name of the team in the filename. For the sample data this is either 'Home' or 'Away'.
    """
    # First:  deal with file headers so that we can get the player names correct
    csvfile = open(tracking_file, "r")  # create a csv file reader
    reader = csv.reader(csvfile)
    teamnamefull = next(reader)[3].lower()
    # construct column names
    jerseys = sorted(
        list(set(x for x in next(reader) if x != ""))
    )  # extract player jersey numbers from second row
    columns = next(reader)
    for i, j in enumerate(
        jerseys
    ):  # create x & y position column headers for each player
        columns[i * 2 + 3] = "{}_{}_x".format(teamname, j)
        columns[i * 2 + 4] = "{}_{}_y".format(teamname, j)
    columns[-2] = "disc_x"  # column headers for the x & y positions of the ball
    columns[-1] = "disc_y"
    # Second: read in tracking data and place into pandas Dataframe
    tracking = pd.read_csv(tracking_file, names=columns, index_col="Frame", skiprows=3)
    return tracking


def to_metric_coordinates(
    data: pd.DataFrame, field_dimen: Tuple[float, float] = (94, 37)
) -> pd.DataFrame:
    """
    Convert positions from Metrica units to meters (with origin at centre circle)
    """
    x_columns = [c for c in data.columns if c[-1].lower() == "x"]
    y_columns = [c for c in data.columns if c[-1].lower() == "y"]
    data[x_columns] = data[x_columns] - field_dimen[0] / 2
    data[y_columns] = -1 * (data[y_columns] - field_dimen[1] / 2)
    """
    ------------ ***NOTE*** ------------
    Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
    I've changed the line above to reflect this. It was originally:
    data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
    ------------ ********** ------------
    """
    return data
