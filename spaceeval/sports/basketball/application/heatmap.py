import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from ...basketball import utils
from PIL import Image
from ..models.BIMOS import BIMOS
from ..models.BMOS import BMOS
import datetime
from ..utils import SportVU_IO as sio

COURT_SIZE = (28, 15)  # Court dimensions in meters
EVENT_LABEL = 1
EVENT_LABELS = {
    'nonevent': 0,
    'pass': 1,
    'catch and pass': 2,
    'handoff catch and pass': 3,
    'catch': 4,
    'handoff pass': 5,
    'handoff catch and handoff pass': 6,
    'catch and handoff pass': 7,
    'handoff catch': 8,
    '2 point shot': 9,
    '3 point shot': 10,
    'turnover': 11
}

def plotCourt():
    with pkg_resources.open_binary(utils, 'nba_court.png') as file:
        img = Image.open(file)
        img = img.copy()

    plt.imshow(img, extent=[0, COURT_SIZE[0], 0, COURT_SIZE[1]], zorder=0)
    plt.xlim(0, COURT_SIZE[0])
    plt.ylim(0, COURT_SIZE[1])

def plot_heat_map_frame(save_path_folder, attValue, data,
                  include_player_velocities = True, BID=True, colorbar = True, title=True
                  ,field_dimen = (COURT_SIZE[0],COURT_SIZE[1]), colormap = "Reds"):

    att_x = []
    att_y = []
    dim_att = []
    for i in range(5):  # 5 attaquants
        x = data[f'x_att{i}'].values[0]
        y = data[f'y_att{i}'].values[0]
        att_x.append(x)
        att_y.append(y)
    
    dim_att = [np.array(att_x), np.array(att_y)]

    def_x = []
    def_y = []
    dim_def = []
    for i in range(5):
        x = data[f'x_def{i}'].values[0]
        y = data[f'y_def{i}'].values[0]
        def_x.append(x)
        def_y.append(y)

    dim_def = [np.array(def_x), np.array(def_y)]
    
    dim_ball = []
    x = data['x_ball'].values[0]
    y = data['y_ball'].values[0]
    dim_ball = np.array([x, y])
    
    fig, ax = plt.subplots(figsize = (6,5))
    fig.subplots_adjust(left = 0.01, bottom = 0.08, right = 0.99, top=0.95)
    plt.text(14,-2,'[m]', ha = 'center')

    ax.imshow(attValue, cmap = colormap, extent =(0, field_dimen[0], 0, field_dimen[1]), alpha=0.9)
    if colorbar:
        plt.colorbar(ax.imshow(attValue, cmap=colormap, extent=(0, field_dimen[0], 0, field_dimen[1]), alpha=0.9))

    plotCourt()

    ax.scatter(*dim_att, s = 100, edgecolor = 'r', c = 'white')
    ax.scatter(*dim_def, s = 100, edgecolor = 'b', c = 'white')
    ax.scatter(*dim_ball, s = 100, edgecolor = 'black', c = 'white')
    ax.scatter(*dim_ball, s = 30, c = 'black')

    if BID and data['ball_holder'].values[0] > 0:
        bid_idx = int(data['ball_holder'].values[0] - 1)
        ax.scatter(
            data[f'x_att{bid_idx}'].values[0], 
            data[f'y_att{bid_idx}'].values[0], 
            s=17, facecolors='none', edgecolors='black'
            )
    
    if include_player_velocities:
        for i in range(5):
            plt.quiver(
                data[f'x_att{i}'].values[0], 
                data[f'y_att{i}'].values[0], 
                data[f'vx_att{i}'].values[0],
                data[f'vy_att{i}'].values[0], 
                angles='xy', scale_units='xy', scale=1, color='black'
                )
            
        for i in range(5):
            plt.quiver(
                data[f'x_def{i}'].values[0], 
                data[f'y_def{i}'].values[0], 
                data[f'vx_def{i}'].values[0],
                data[f'vy_def{i}'].values[0], 
                angles='xy', scale_units='xy', scale=1, color='black'
                )
        
        # Annotate defending players
        for i in range(5):
            player_id = int(data[f'IDdef_{i}'].values[0]) if f'IDdef_{i}' in data.columns else i
            plt.text(data[f'x_def{i}'].values[0],data[f'y_def{i}'].values[0], f'{player_id}', fontsize=8)
    

    if title:
        plt.title(f"Game {data['gameID'].values[0]} - Event {data['eventid'].values[0]} - Frame {data['f_id'].values[0]}")

    plt.show()

    
    if 'gameID' in data.columns:
        game_id = data['gameID'].values[0]
    else:
        game_id = 'unknown'
    
    if 'eventid' in data.columns:
        event_id = data['eventid'].values[0]
    else:
        event_id = 'unknown'
    
    if 'f_id' in data.columns:
        frame_id = data['f_id'].values[0]
    else:
        frame_id = 'unknown'
    
    filename = f"heatmap_game_{game_id}_event_{event_id}_frame_{frame_id}.png"
    save_path = f"{save_path_folder}/{filename}"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_heat_map_sequence(model, data, heatmap=True, 
    EVENT=True, JERSEY=True, BID=False, axis=False, title=True, field_dimen=(COURT_SIZE[0],COURT_SIZE[1])):
    """
    Plots animation for a specific scene.

    Parameters:
    -----------
        game_id, s_id, f_id: IDs for the game, scene, and frame.
        params: Parameters for pitch control calculation.
        fit_params: Parameters for tau_true - tau_exp distribution fitting.
        integral_xmin: Integration limit for pitch control.
        version: One of "BMOS", "BIMOS", "PPCF", or "PBCF".
        heatmap: If True, displays heatmap.
        EVENT: If True, displays event labels.
        JERSEY: If Ture, displays jersey numbers
        BID: If True, highlights the player in ball possession.
        axis: If True, shows axis ticks and labels.
        title: If True, adds a title to the plot.
    -----------
    """

    def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        return keys[0]
    
    def extract_date_info(gamename):
        parts = gamename.split('_')

        day = int(parts[1])
        month = int(parts[0])
        year = int(parts[2])
        
        suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        day_formatted = f"{day}{suffix}"
        
        month_name = datetime.date(1900, month, 1).strftime('%b')
        
        return day_formatted, month_name, str(year)
    
    game_id = data['gameID'].values[0]
    s_id = data['eventid'].values[0]

    # Obtain date info
    gamename = data['gamename'].values[0]
    date, month, year = extract_date_info(gamename)

    # Obtain team info
    team_id_O = data['team_O'].values[0]
    team_id_D = data['team_D'].values[0]
    team_name_O = sio.load_team_name(team_id_O)
    team_name_D = sio.load_team_name(team_id_D)

    # Precalculate frame data
    precalculated_data = []
    for i in range(len(data)):
        row_cumput = data.iloc[[i]].copy()

        att_x = []
        att_y = []
        dim_att = []
        for i in range(5):  # 5 attaquants
            x = row_cumput[f'x_att{i}'].values[0]
            y = row_cumput[f'y_att{i}'].values[0]
            att_x.append(x)
            att_y.append(y)

        dim_att = [np.array(att_x), np.array(att_y)]

        def_x = []
        def_y = []
        dim_def = []
        for i in range(5):
            x = row_cumput[f'x_def{i}'].values[0]
            y = row_cumput[f'y_def{i}'].values[0]
            def_x.append(x)
            def_y.append(y)

        dim_def = [np.array(def_x), np.array(def_y)]
        
        dim_ball = []
        x = row_cumput['x_ball'].values[0]
        y = row_cumput['y_ball'].values[0]
        dim_ball = np.array([x, y])

        frame_info = {
            'dim_att': dim_att,
            'dim_def': dim_def,
            'dim_ball': dim_ball
        }

        if model == "BMOS":
            frame_info['attValue'] = BMOS(row_cumput).values
        
        if model == "BIMOS":
            frame_info['attValue'] = BIMOS(row_cumput).values

        if EVENT:
            frame_info['event_label'] = row_cumput['event_label'].values[0]

        if JERSEY:
            frame_info['jersey_number'] = row_cumput.filter(regex='^jersey_', axis=1).iloc[0].tolist()
            
        precalculated_data.append(frame_info)

    # Animation
    fig, ax = plt.subplots()

    def animate(f_id):
        ax.clear()
        frame_info = precalculated_data[f_id]

        # Plot heatmap
        if heatmap:
            ax.imshow(frame_info['attValue'], cmap='Reds', vmin=0., vmax=1., 
                    extent=(0, field_dimen[0], 0, field_dimen[1]), alpha=0.9)
            
        # Plot players and ball
        ax.scatter(*frame_info['dim_att'],  s=100, edgecolor ='r', c = "white")
        ax.scatter(*frame_info['dim_def'], s=100, edgecolor='b', c = "white")
        ax.scatter(*frame_info['dim_ball'], s=30, c = "black")

        # Highlight ball possessor
        if BID and data['ball_holder'].values[0] > 0:
            bid_idx = int(data['ball_holder'].values[0] - 1)
            ax.scatter(
                data[f'x_att{bid_idx}'].values[0], 
                data[f'y_att{bid_idx}'].values[0],
                s=17, facecolors='none', edgecolors='black'
                )

        # Add event labels
        if EVENT:
            ax.text(*frame_info['dim_ball'], f"{frame_info['event_label']}", fontsize=8)

        # Add jersey numbers
        if JERSEY:
            for i in np.arange(10):
                jersey_numbers = int(frame_info["jersey_number"][i])
                if i < 5:
                    x, y = frame_info['dim_att'][0][i], frame_info['dim_att'][1][i]
                else:
                    x, y = frame_info['dim_def'][0][i-5], frame_info['dim_def'][1][i-5]
                ax.text(x, y, 
                        f'{jersey_numbers}', 
                        fontsize=8, horizontalalignment='center', verticalalignment='center',)

        # Add titles
        fid_str = str(f_id).zfill(3)
        if not title:
            key = get_key_from_value(EVENT_LABELS, data[len(data)-1][EVENT_LABEL])
            ax.set_title(f'Game {game_id} - Event {s_id} - Frame {fid_str} - {key}')
        else:
            ax.set_title('')
            ax.text(0.08, 1.025, team_name_O, color='red', fontsize=12, ha='center', transform=ax.transAxes)
            ax.text(0.17, 1.025, 'vs.', color='black', fontsize=12, ha='center', transform=ax.transAxes)
            ax.text(0.25, 1.025, team_name_D, color='blue', fontsize=12, ha='center', transform=ax.transAxes)
            ax.text(0.65, 1.025, date + ' ' + month + '. ' +  year + ' - Frame ' + fid_str, 
                    color='black', fontsize=12, ha='center', transform=ax.transAxes)

        if not axis:
            plt.xticks([])
            plt.yticks([])  

        plotCourt()

    ani = animation.FuncAnimation(fig, animate, frames=len(data), interval=100)
    plt.show()