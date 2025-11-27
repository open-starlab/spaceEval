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
