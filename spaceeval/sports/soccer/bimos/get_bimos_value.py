
from .c_obso_repo import Metrica_IO as mio
from .c_obso_repo import Metrica_Velocities as mvel
from .c_obso_repo import Metrica_PitchControl as mpc
from .c_obso_repo import Metrica_EPV as mepv

import pandas as pd 
import numpy as np
from tqdm import tqdm

import importlib.resources as pkg_resources
import spaceeval.sports.soccer.obso.c_obso_repo as c_repo

from .c_obso_repo import obso_player as obs
from .bimos_repo.cal_pbcf import generate_pbcf_for_event

def calculate_bimos_fc(Metrica_df, tracking_home, tracking_away):
    """
    Calculate player-based Offensive Ball Space Occupancy (OBSO) metrics for events.

    Parameters
    ----------
    Metrica_df : pd.DataFrame
        Event data in Metrica format (with 'Start Frame', 'Team', 'Period', etc.)

    tracking_home : pd.DataFrame
        Home team tracking data with player coordinates and ball positions.

    tracking_away : pd.DataFrame
        Away team tracking data with player coordinates and ball positions.

    Returns
    -------
    home_obso : np.ndarray
        OBSO for home team players.

    away_obso : np.ndarray
        OBSO for away team players.

    home_onball_obso : np.ndarray
        OBSO for home team players currently in possession.

    away_onball_obso : np.ndarray
        OBSO for away team players currently in possession.

    PBCF_dict : dict
        Pitch control field (PBCF) per event.
    """
    
    Metrica_df = Metrica_df.head(30)

    tracking_len = min(len(tracking_home), len(tracking_away))
    #drop the Merica_df_row where Start Frame > tracking_len
    Metrica_df = Metrica_df[Metrica_df['Start Frame'] < tracking_len].reset_index(drop=True)

    # check 'Home' team in tracking and event data
    Metrica_df = obs.check_home_away_event(Metrica_df, tracking_home, tracking_away)
    # delete last event because this event is 'time up' event
    Metrica_df = Metrica_df[:-1]

    # filter:Savitzky-Golay
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True) 
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    # set parameter
    params = mpc.default_model_params()
    GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]

    # load control and transition model
    with pkg_resources.path(c_repo, 'EPV_grid.csv') as csv_path:
        EPV = mepv.load_EPV_grid(csv_path)
        EPV = EPV / np.max(EPV)
    with pkg_resources.path(c_repo, 'Transition_gauss.csv') as csv_path:
        Trans_df = pd.read_csv(csv_path, header=None)
        Trans = np.array((Trans_df))
        Trans = Trans / np.max(Trans)

    # set OBSO data
    obso = np.zeros((len(Metrica_df), 32, 50))
    PBCF_dict = {}
    for event_num, frame in tqdm(enumerate(Metrica_df['Start Frame']), total=len(Metrica_df['Start Frame']), desc="Calculating PBCF", leave=False):
        if Metrica_df['Team'].loc[event_num]=='Home':
            # check attack direction 1st half or 2nd half
            if Metrica_df.loc[event_num]['Period']==1:
                direction = mio.find_playing_direction(tracking_home[tracking_home['Period']==1], 'Home')
            elif Metrica_df.loc[event_num]['Period']==2:
                direction = mio.find_playing_direction(tracking_home[tracking_home['Period']==2], 'Home')
            PBCF, _, _, _ = generate_pbcf_for_event(event_num, Metrica_df, tracking_home, tracking_away, params, GK_numbers, offsides=True)
            PBCF_dict[event_num] = PBCF
        elif Metrica_df['Team'].loc[event_num]=='Away': 
            # check attack direction 1st half or 2nd half
            if Metrica_df.loc[event_num]['Period']==1:
                direction = mio.find_playing_direction(tracking_away[tracking_away['Period']==1], 'Away')
            elif Metrica_df.loc[event_num]['Period']==2:
                direction = mio.find_playing_direction(tracking_away[tracking_away['Period']==2], 'Away')
            PBCF, _, _, _ = generate_pbcf_for_event(event_num, Metrica_df, tracking_home, tracking_away, params, GK_numbers, offsides=True)
            PBCF_dict[event_num] = PBCF
        else:
            obso[event_num] = np.zeros((32, 50))
            PBCF_dict[event_num] = np.zeros((32, 50))
            continue
        obso[event_num], _ = obs.calc_obso(PBCF, Trans, EPV, tracking_home.loc[frame], attack_direction=direction)

    home_obso, away_obso = obs.calc_player_evaluate_match(obso, Metrica_df, tracking_home, tracking_away)

    # calculate onball obso
    home_onball_obso, away_onball_obso = obs.calc_onball_obso(Metrica_df, tracking_home, tracking_away, home_obso, away_obso)
    # remove offside player
    home_obso, away_obso = obs.remove_offside_obso(Metrica_df, tracking_home, tracking_away, home_obso, away_obso)

    return home_obso, away_obso, home_onball_obso, away_onball_obso, PBCF_dict

