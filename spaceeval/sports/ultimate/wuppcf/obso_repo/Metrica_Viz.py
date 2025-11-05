#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:10:58 2020

Module for visualising Metrica tracking and event data

Data can be found at: https://github.com/metrica-sports/sample-data

UPDATE for tutorial 4: plot_pitchcontrol_for_event no longer requires 'xgrid' and 'ygrid' as inputs. 

@author: Laurie Shaw (@EightyFivePoint)
"""

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
from tqdm import tqdm


def plot_pitch(field_dimen=(94.0, 37.0), linewidth=2, markersize=20, grid_size=2):
    """plot_pitch

    Plots a soccer pitch. All distance units converted to meters.

    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20

    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    fig, ax = plt.subplots(figsize=(12, 8))  # create a figure
    # decide what color we want the field to be. Default is green, but can also choose white
    field_color = "white"
    line_color = "k"
    # ALL DIMENSIONS IN m
    border_dimen = (3, 3)  # include a border arround of the field of width 3m
    half_pitch_length = field_dimen[0] / 2  # length of half pitch
    half_pitch_width = field_dimen[1] / 2  # width of half pitch
    if field_dimen[0] == 94.0:
        end_zone_line = (
            half_pitch_length - 18.0
        )  # 18.0m is the depth of an ultimate end zone
    elif field_dimen[0] == 109.73:
        end_zone_line = (
            half_pitch_length - 18.29
        )  # 18.29m is the depth of end zone of UFA
    signs = [-1, 1]
    # plot end zones
    ax.plot(
        [-1 * end_zone_line, -1 * end_zone_line],
        [-half_pitch_width, half_pitch_width],
        line_color,
        linewidth=linewidth,
    )
    ax.plot(
        [end_zone_line, end_zone_line],
        [-half_pitch_width, half_pitch_width],
        line_color,
        linewidth=linewidth,
    )
    for s in signs:  # plots each line seperately
        # plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            line_color,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            line_color,
            linewidth=linewidth,
        )

    # remove axis labels and ticks
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2 + border_dimen[0]
    ymax = field_dimen[1] / 2 + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)
    return fig, ax


def save_match_clip_OBSO(
    hometeam,
    awayteam,
    wUPPCF,
    fpath,
    fname="clip_test",
    figax=None,
    frames_per_second=25,
    team_colors=("b", "r"),
    field_dimen=(94.0, 37.0),
    include_player_velocities=False,
    PlayerMarkerSize=10,
    PlayerAlpha=0.7,
    annotate=False,
    vmin=0,
    vmax=1,
    colorbar=False,
    cm=cm.bwr_r,
    grid_size=2.0,
):
    assert np.all(
        hometeam.index == awayteam.index
    ), "Home and away team Dataframe indices must be the same"
    index = hometeam.index
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(
        title="Tracking Data", artist="Matplotlib", comment="Metrica tracking data clip"
    )
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = f"{fpath}/{fname}.mp4"

    if figax is None:
        fig, ax = plot_pitch(field_dimen=field_dimen, grid_size=grid_size)
    else:
        fig, ax = figax
    fig.set_tight_layout(True)

    # 初期化
    player_objs = []
    disc_objs = []
    text_objs = []
    quiver_objs = []
    for team, color in zip(
        [hometeam.loc[index[0]], awayteam.loc[index[0]]], team_colors
    ):
        x_columns = [c for c in team.keys() if c[-2:].lower() == "_x" and c != "disc_x"]
        y_columns = [c for c in team.keys() if c[-2:].lower() == "_y" and c != "disc_y"]
        player_objs.append(
            ax.plot(
                team[x_columns] * grid_size,
                team[y_columns] * grid_size,
                color + "o",
                markersize=PlayerMarkerSize,
                alpha=PlayerAlpha,
            )[0]
        )
        disc_objs.append(
            ax.plot(
                team["disc_x"] * grid_size,
                team["disc_y"] * grid_size,
                "ko",
                markersize=6,
                alpha=1.0,
                linewidth=0,
            )[0]
        )
        if include_player_velocities:
            vx_columns = ["{}_vx".format(c[:-2]) for c in x_columns]
            vy_columns = ["{}_vy".format(c[:-2]) for c in y_columns]
            quiver_objs.append(
                ax.quiver(
                    team[x_columns] * grid_size,
                    team[y_columns] * grid_size,
                    team[vx_columns] * grid_size,
                    team[vy_columns] * grid_size,
                    color=color,
                    scale_units="inches",
                    scale=10.0,
                    width=0.0015,
                    headlength=5,
                    headwidth=3,
                    alpha=PlayerAlpha,
                )
            )
        if annotate:
            for x, y in zip(x_columns, y_columns):
                if not np.isnan(team[x]) and not np.isnan(team[y]):
                    text_objs.append(
                        ax.text(
                            team[x] * grid_size + 0.25,
                            team[y] * grid_size + 0.25,
                            x.split("_")[1],
                            fontsize=10,
                            color=color,
                        )
                    )
    cmap = cm
    obs_map = ax.imshow(
        np.flipud(wUPPCF[0]),
        extent=(
            -field_dimen[0] / 2,
            field_dimen[0] / 2,
            -field_dimen[1] / 2,
            field_dimen[1] / 2,
        ),
        # interpolation="spline36",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        alpha=0.7,
    )
    if colorbar:
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        fig.colorbar(obs_map, cax=cax)

    with writer.saving(fig, fname, 100):
        for frame, i in tqdm(
            enumerate(index), total=len(index), desc="Generating clip"
        ):
            for team, color, player_obj, disc_obj, quiver_obj in zip(
                [hometeam.loc[i], awayteam.loc[i]],
                team_colors,
                player_objs,
                disc_objs,
                quiver_objs,
            ):
                x_columns = [
                    c for c in team.keys() if c[-2:].lower() == "_x" and c != "disc_x"
                ]
                y_columns = [
                    c for c in team.keys() if c[-2:].lower() == "_y" and c != "disc_y"
                ]
                player_obj.set_data(
                    team[x_columns] * grid_size, team[y_columns] * grid_size
                )
                disc_obj.set_data(
                    [team["disc_x"] * grid_size], [team["disc_y"] * grid_size]
                )
                if include_player_velocities:
                    vx_columns = ["{}_vx".format(c[:-2]) for c in x_columns]
                    vy_columns = ["{}_vy".format(c[:-2]) for c in y_columns]
                    quiver_obj.set_offsets(
                        np.c_[team[x_columns] * grid_size, team[y_columns] * grid_size]
                    )
                    quiver_obj.set_UVC(
                        team[vx_columns] * grid_size, team[vy_columns] * grid_size
                    )
                if annotate:
                    if team.index[2].split("_")[0] == "Home":
                        for text_obj, x, y in zip(text_objs, x_columns, y_columns):
                            if not np.isnan(team[x]) and not np.isnan(team[y]):
                                text_obj.set_position(
                                    (
                                        team[x] * grid_size + 0.25,
                                        team[y] * grid_size + 0.25,
                                    )
                                )
                            if float(x[-3]) == team["Selected"]:
                                text_obj.set_color("yellow")
                                text_obj.set_fontsize(15)
                            else:
                                text_obj.set_color("blue")
                                text_obj.set_fontsize(10)
                    else:
                        for text_obj, x, y in zip(text_objs[7:], x_columns, y_columns):
                            if not np.isnan(team[x]) and not np.isnan(team[y]):
                                text_obj.set_position(
                                    (
                                        team[x] * grid_size + 0.25,
                                        team[y] * grid_size + 0.25,
                                    )
                                )
                            if float(x[-3]) == team["Selected"]:
                                text_obj.set_color("green")
                                text_obj.set_fontsize(15)
                            else:
                                text_obj.set_color("red")
                                text_obj.set_fontsize(10)
            obs_map.set_data(np.flipud(wUPPCF[frame]))
            frame_minute = int(team["Time [s]"] / 60.0)
            frame_second = (team["Time [s]"] / 60.0 - frame_minute) * 60.0
            timestring = "%1.2f, %d" % (frame_second, frame)
            time_text = ax.text(-2.5, field_dimen[1] / 2 + 1.0, timestring, fontsize=14)
            writer.grab_frame()
            time_text.remove()

    plt.clf()
    plt.close(fig)
