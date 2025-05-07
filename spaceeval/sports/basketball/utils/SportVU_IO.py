from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import importlib.resources as pkg_ressources
from ...basketball import utils
import os

# event data index
SCORE = 2
CALC_FID = 6
LAST_CHOICE = 7
CALC_POS = slice(8, 10)

LAST_CHOICE_LABELS = {
    'pass': 0,    # pass-to-score sequence
    'dribble': 1  # dribble-to-score sequence
}

def load_tracking_data(game_id, onball=False):
    game_str = str(game_id).zfill(3)
    if onball:
        file_name = f"attackDataset_game{game_str}.mat"
        current_path = os.getcwd()
        onball_dir = "onball_scoreDataset"
        file_path = os.path.join(current_path, onball_dir, file_name)
        with open(file_path, 'rb') as file:
            t_data = loadmat(file)['data'][0]
    return t_data

def load_event_data(game_id, onball=False):
    if onball:
        file_name = "onballevents_dataset.mat"
        with pkg_ressources.open_binary(utils, file_name) as file:
            e_data = loadmat(file)['event'][0][game_id - 1][0]

    else:
        file_name = "allevents_dataset.mat"
        with pkg_ressources.open_binary(utils, file_name) as file:
            e_data = loadmat(file)['event'][0][game_id - 1][0]
    return e_data

def load_team_name(team_id):
    file_name = "id_team.csv"
    with pkg_ressources.open_binary(utils, file_name) as file:
           team_data = pd.read_csv(file)
    
    team_info = team_data.loc[team_data['team_id'] == team_id]
    team_name = team_info.iloc[0]['team_2']
    return team_name

def get_pos_id(pos):
    FIELD_DIMS = (28, 15)
    yid, xid = 14 - math.floor(pos[1]), math.floor(pos[0])
    yid = max(0, min(yid, FIELD_DIMS[1] - 1))
    xid = max(0, min(xid, FIELD_DIMS[0] - 1))
    print(yid,xid)
    return [yid, xid]

def make_transitionmodel_for_event(data, field_dimen = (28.,15.)):
    """
    Apply the transition model to the current pitch situation.
    """

    file_name = "transitionmodel.csv"
    with pkg_ressources.open_binary(utils, file_name) as file:
          transitionmodel = np.array(pd.read_csv(file, header=None))

    print(transitionmodel)

    x_array = np.arange(0, field_dimen[0], 1) 
    print(x_array) 

    y_array = np.arange(field_dimen[1]-1, -1, -1)
    print(y_array)

    transition = np.empty((15, 28))
    dim_ball_id = get_pos_id([data['x_ball'].values[0], data['y_ball'].values[0]])
    center_id = [25, 25]

    for iy in y_array:
        for ix in x_array:
            transition[int(iy),int(ix)] = transitionmodel[int(center_id[0] + (iy - dim_ball_id[0])), int(center_id[1] + (ix - dim_ball_id[1]))]

    return transition