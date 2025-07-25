import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from ...basketball import utils
from PIL import Image
from ..models.BIMOS import BIMOS
from ..models.BMOS import BMOS
import datetime
from ..utils import SportVU_IO as sio
import glob
import cv2
from joblib import Parallel, delayed

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

def plot_heat_map_frame(data, save_path_folder, attValue,
                  include_player_velocities = True, BID=True, colorbar = True, title=True
                  ,field_dimen = (COURT_SIZE[0],COURT_SIZE[1]), colormap = "Reds"):

    att_x = []
    att_y = []
    dim_att = []
    for a in range(5):  # 5 attaquants
        x = data[f'x_att{a}'].values[0]
        y = data[f'y_att{a}'].values[0]
        att_x.append(x)
        att_y.append(y)
    
    dim_att = [np.array(att_x), np.array(att_y)]

    def_x = []
    def_y = []
    dim_def = []
    for b in range(5):
        x = data[f'x_def{b}'].values[0]
        y = data[f'y_def{b}'].values[0]
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

    if BID and data['ball_holder'].values[0] > 0 and data['ball_holder'].values[0] < 9:
        bid_idx = int(data['ball_holder'].values[0] - 1)
        ax.scatter(
            data[f'x_att{bid_idx}'].values[0], 
            data[f'y_att{bid_idx}'].values[0], 
            s=17, facecolors='none', edgecolors='black'
            )
    
    if include_player_velocities:
        for c in range(5):
            plt.quiver(
                data[f'x_att{c}'].values[0], 
                data[f'y_att{c}'].values[0], 
                data[f'vx_att{c}'].values[0],
                data[f'vy_att{c}'].values[0], 
                angles='xy', scale_units='xy', scale=1, color='black'
                )
            
        for d in range(5):
            plt.quiver(
                data[f'x_def{d}'].values[0], 
                data[f'y_def{d}'].values[0], 
                data[f'vx_def{d}'].values[0],
                data[f'vy_def{d}'].values[0], 
                angles='xy', scale_units='xy', scale=1, color='black'
                )
    
    if title:
        plt.title(f"Game {data['gameID'].values[0]} - Attack {data['attackid'].values[0]} - Frame {data['f_id'].values[0]}")

    
    if 'gameID' in data.columns:
        game_id = data['gameID'].values[0]
    else:
        game_id = 'unknown'
    
    if 'attackid' in data.columns:
        attackid = data['attackid'].values[0]
    else:
        attackid = 'unknown'
    
    if 'f_id' in data.columns:
        frame_id = data['f_id'].values[0]
    else:
        frame_id = 'unknown'
    
    filename = f"heatmap_game_{game_id}_attack_{attackid}_frame_{frame_id}.png"
    save_path = f"{save_path_folder}/{filename}"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def movie_from_images(image_files, output_file, frames_with_heatmap, fps=10, heatmap_duration_multiplier=10):
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_indices = {}
    for i, file in enumerate(image_files):
        frame_number = int(file.split('_')[-1].split('.')[0])
        frame_indices[file] = frame_number

    for file in image_files:
        img = cv2.imread(file)
        frame_number = frame_indices[file]
        
        if frame_number in frames_with_heatmap:
            for _ in range(heatmap_duration_multiplier):
                out.write(img)
        else:
            out.write(img)

    out.release()

    # Clean up by deleting image files after creating the movie
    print("Video created at:", output_file)

    if os.path.exists("figures"):
        for file in glob.glob("figures/*"):
            os.remove(file)
        os.rmdir("figures")
        print("Figures directory removed")


def extract_date_info(gamename):
    parts = gamename.split('_')

    day = int(parts[1])
    month = int(parts[0])
    year = int(parts[2])
    
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    day_formatted = f"{day}{suffix}"
    
    month_name = datetime.date(1900, month, 1).strftime('%b')
    
    return day_formatted, month_name, str(year)

def process_frame(i, data, model, save_path_folder, heatmap, only_event, EVENT, JERSEY, BID, axis, title, field_dimen, team_name_O, team_name_D, date, month, year, game_id, s_id):
    row_cumput = data.iloc[[i]].copy()

    att_x = [row_cumput[f'x_att{j}'].values[0] for j in range(5)]
    att_y = [row_cumput[f'y_att{j}'].values[0] for j in range(5)]
    dim_att = [np.array(att_x), np.array(att_y)]

    def_x = [row_cumput[f'x_def{j}'].values[0] for j in range(5)]
    def_y = [row_cumput[f'y_def{j}'].values[0] for j in range(5)]
    dim_def = [np.array(def_x), np.array(def_y)]

    dim_ball = np.array([row_cumput['x_ball'].values[0], row_cumput['y_ball'].values[0]])

    frame_info = {
        'dim_att': dim_att,
        'dim_def': dim_def,
        'dim_ball': dim_ball
    }

    if only_event:
        if row_cumput['event_label'].values[0] not in [0, 4, 8]:
            if model == "BIMOS":
                frame_info['attValue'] = BIMOS(row_cumput).values
            else:
                frame_info['attValue'] = BMOS(row_cumput).values
        else:
            frame_info['attValue'] = [[0] * 28 for _ in range(15)]
    else:
        if row_cumput['ball_hold'].values[0] == 0:
            frame_info['attValue'] = [[0] * 28 for _ in range(15)]
        else:
            if model == "BIMOS":
                frame_info['attValue'] = BIMOS(row_cumput).values
            else:
                frame_info['attValue'] = BMOS(row_cumput).values

    if EVENT:
        frame_info['event_label'] = row_cumput['event_label'].values[0]

    if JERSEY:
        frame_info['jersey_number'] = row_cumput.filter(regex='^jersey_', axis=1).iloc[0].tolist()

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=0.01, bottom=0.08, right=0.99, top=0.95)
    plt.text(14, -2, '[m]', ha='center')

    if heatmap:
        ax.imshow(frame_info['attValue'], cmap='Reds', vmin=0.15, vmax=1.,
                  extent=(0, field_dimen[0], 0, field_dimen[1]), alpha=0.8)

    ax.scatter(*frame_info['dim_att'], s=100, edgecolor='r', c="white")
    ax.scatter(*frame_info['dim_def'], s=100, edgecolor='b', c="white")
    ax.scatter(*frame_info['dim_ball'], s=30, c="black")

    if BID and row_cumput['ball_holder'].values[0] > 0 and row_cumput['ball_holder'].values[0] < 9:
        bid_idx = int(row_cumput['ball_holder'].values[0] - 1)
        ax.scatter(
            row_cumput[f'x_att{bid_idx}'].values[0],
            row_cumput[f'y_att{bid_idx}'].values[0],
            s=60, facecolors='none', edgecolors='black'
        )

    if EVENT and 'event_label' in frame_info:
        ax.text(*frame_info['dim_ball'], f"{frame_info['event_label']}", fontsize=8)

    if JERSEY and 'jersey_number' in frame_info:
        for j in range(10):
            jersey_number = int(frame_info["jersey_number"][j])
            if j < 5:
                x, y = frame_info['dim_att'][0][j], frame_info['dim_att'][1][j]
            else:
                x, y = frame_info['dim_def'][0][j-5], frame_info['dim_def'][1][j-5]
            ax.text(x, y, f'{jersey_number}', fontsize=8, ha='center', va='center')

    fid_str = str(i).zfill(3)
    if title:
        ax.set_title('')
        ax.text(0.08, 1.025, team_name_O, color='red', fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.17, 1.025, 'vs.', color='black', fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.25, 1.025, team_name_D, color='blue', fontsize=12, ha='center', transform=ax.transAxes)
        ax.text(0.65, 1.025, f"{date} {month}. {year} - Frame {fid_str}",
                color='black', fontsize=12, ha='center', transform=ax.transAxes)
    else:
        ax.set_title(f'Game {game_id} - Attack {s_id} - Frame {fid_str}')

    if not axis:
        plt.xticks([])
        plt.yticks([])

    plotCourt()

    if not os.path.exists("figures"):
        os.makedirs("figures")
    img_path = f"figures/frame_{game_id}_{s_id}_{fid_str}.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

def plot_heat_map_sequence(model, data, save_path_folder, heatmap=True, only_event=True,
    EVENT=True, JERSEY=True, BID=False, axis=False, title=True, field_dimen=(COURT_SIZE[0],COURT_SIZE[1])):
    """
    Plots animation for a specific scene, using parallel processing for frames.
    """
    gamename = data['gamename'].values[0]
    date, month, year = extract_date_info(gamename)
    game_id = data['game'].values[0]
    s_id = data['attackid'].values[0]
    team_id_O = data['team_O'].values[0]
    team_id_D = data['team_D'].values[0]
    team_name_O = sio.load_team_name(team_id_O)
    team_name_D = sio.load_team_name(team_id_D)

    frames_with_heatmap = []
    for i in range(len(data)):
        row_cumput = data.iloc[[i]].copy()
        if only_event and row_cumput['event_label'].values[0] not in [0, 4, 8]:
            frames_with_heatmap.append(i)

    # Barre de chargement tqdm sur le traitement des frames
    indices = list(range(len(data)))
    image_files = Parallel(n_jobs=-1)(
        delayed(process_frame)(
            i, data, model, save_path_folder, heatmap, only_event, EVENT, JERSEY, BID, axis, title, field_dimen,
            team_name_O, team_name_D, date, month, year, game_id, s_id
        )
        for i in tqdm(indices, desc="computation progress", unit="frame")
    )

    image_files = sorted(glob.glob(f"figures/frame_{game_id}_{s_id}_*.png"))
    print(image_files)
    video_name = f"{save_path_folder}/space_evaluation_{game_id}_{s_id}.mp4"
    movie_from_images(image_files=image_files, output_file=video_name, frames_with_heatmap = frames_with_heatmap)

