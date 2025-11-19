#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:52:19 2020

Module for calculating a Pitch Control surface using MetricaSports's tracking & event data.

Pitch control (at a given location on the field) is the probability that a team will gain
possession if the ball is moved to that location on the field.

Methdology is described in "Off the ball scoring opportunities" by William Spearman:
http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf

GitHub repo for this code can be found here:
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Data can be found at: https://github.com/metrica-sports/sample-data

Functions
----------

calculate_pitch_control_at_target(): calculate the pitch control probability for the attacking and defending teams at a specified target position on the ball.

generate_pitch_control_for_event(): this function evaluates pitch control surface over the entire field at the moment
of the given event (determined by the index of the event passed as an input)

Classes
---------

The 'player' class collects and stores trajectory information for each player required by the pitch control calculations.

@author: Laurie Shaw (@EightyFivePoint)

"""

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def initialise_players(
    attacking_team: pd.DataFrame,
    defending_team: pd.DataFrame,
    attacking_teamname: str,
    defending_teamname: str,
    params: dict,
    removed_player,
    field_dimen: Tuple[float, float] = (94.0, 37.0),
):
    """
    initialise_players(team,teamname,params)

    create a list of player objects that holds their positions and velocities from the tracking data dataframe

    Parameters
    -----------

    team: row (i.e. instant) of either the home or away team tracking Dataframe
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )

    Returns
    -----------

    team_players: list of player objects for the team at at given instant

    """
    # get player ids
    attacking_player_ids = np.arange(1, 8)
    defending_player_ids = np.arange(1, 8)

    # remove attacking player near the disc
    defending_removed_player = None
    if removed_player is not None:
        # Convert Series to int safely
        removed_player_id = str(
            int(removed_player.iloc[0])
            if hasattr(removed_player, "iloc")
            else int(removed_player)
        )

        attacking_player_ids = np.delete(
            attacking_player_ids,
            np.where(attacking_player_ids == int(removed_player_id)),
        )
        # remove the nearest difending player within 3 meters of the attacking player to be removed
        disc_holder_loc = np.array(
            [
                attacking_team[_]
                for _ in attacking_team.keys()
                if _[5] == str(removed_player_id) and len(_) == 8
            ]
        )
        if len(disc_holder_loc) != 0:
            disc_holder_loc = np.array(
                [
                    disc_holder_loc[0] * 2 + field_dimen[0],
                    -1 * disc_holder_loc[1] * 2 + field_dimen[1],
                ]
            )
        dist_min = 100
        for i in defending_player_ids:
            defending_player_loc = np.array(
                [
                    defending_team[_]
                    for _ in defending_team.keys()
                    if _[5] == str(i) and len(_) == 8
                ]
            )
            defending_player_loc = np.array(
                [
                    defending_player_loc[0] * 2 + field_dimen[0],
                    -1 * defending_player_loc[1] * 2 + field_dimen[1],
                ]
            )
            if len(disc_holder_loc) == 0:
                continue
            dist = np.linalg.norm(disc_holder_loc - defending_player_loc)
            dist_min = min(dist, dist_min)
            if dist == dist_min:
                temp = i
        if dist_min <= 3.0:
            defending_removed_player = temp
            defending_player_ids = defending_player_ids[
                defending_player_ids != defending_removed_player
            ]

    # create list
    attacking_players = []
    for p in attacking_player_ids:
        # create a player object for player_id 'p'
        attacking_player = player(
            p, attacking_team, defending_team, attacking_teamname, params
        )
        if attacking_player.inframe:
            attacking_players.append(attacking_player)
    defending_players = []
    for p in defending_player_ids:
        # create a player object for player_id 'p'
        defending_player = player(
            p, defending_team, attacking_team, defending_teamname, params
        )
        if defending_player.inframe:
            defending_players.append(defending_player)
    return attacking_players, defending_players, defending_removed_player


class player(object):
    """
    player() class

    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player

    __init__ Parameters
    -----------
    pid: id (jersey number) of player
    team: row of tracking data for team
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )


    methods include:
    -----------
    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept

    """

    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(
        self,
        pid: int,
        team: pd.DataFrame,
        opponent_team: pd.DataFrame,
        teamname: str,
        params: dict,
    ):
        self.id = pid
        self.teamname = teamname
        self.playername = "%s_%s_" % (teamname, pid)
        self.vmax = params[
            "max_player_speed"
        ]  # player max speed in m/s. Could be individualised
        self.tti_sigma = params[
            "tti_sigma"
        ]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params[
            "lambda_att"
        ]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params[
            "lambda_def"
        ]  # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.get_position(team)
        self.get_velocity(team)
        self.get_reaction_time(team, opponent_team)
        self.UPPCF = 0.0  # initialise this for later
        self.wUPPCF = 0.0  # initialise this for later

    def get_position(self, team: pd.DataFrame) -> None:
        self.position = np.array(
            [team[self.playername + "x"], team[self.playername + "y"]]
        )
        self.inframe = not np.any(np.isnan(self.position))

    def get_velocity(self, team: pd.DataFrame) -> None:
        self.velocity = np.array(
            [team[self.playername + "vx"], team[self.playername + "vy"]]
        )
        if np.any(np.isnan(self.velocity)):
            self.velocity = np.array([0.0, 0.0])

    def get_reaction_time(
        self, team: pd.DataFrame, opponent_team: pd.DataFrame
    ) -> None:
        disc_pos = np.array([team["disc_x"], team["disc_y"]])
        direction_to_disc = disc_pos - self.position
        direction_velocity = self.velocity

        # Handle cases where vectors have zero magnitude
        disc_norm = np.linalg.norm(direction_to_disc)
        velocity_norm = np.linalg.norm(direction_velocity)

        if disc_norm < 1e-10 or velocity_norm < 1e-10:
            # If player is at disc position or has no velocity, use default reaction time
            self.reaction_time = 0.1
            return

        # angle from direction_to_disc to direction_velocity
        dot_product = np.dot(direction_to_disc, direction_velocity)
        cos_theta = dot_product / (disc_norm * velocity_norm)
        # Clamp to [-1, 1] to avoid numerical errors in arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if self.teamname == "Away":
            direction_to_offense = (
                np.array(
                    [
                        opponent_team[self.playername.replace("Away", "Home") + "x"],
                        opponent_team[self.playername.replace("Away", "Home") + "y"],
                    ]
                )
                - self.position
            )

            offense_norm = np.linalg.norm(direction_to_offense)
            if offense_norm > 1e-10:
                # angle from direction_to_disc to direction_to_offense
                dot_product_defense = np.dot(direction_to_disc, direction_to_offense)
                cos_defense_theta = dot_product_defense / (disc_norm * offense_norm)
                cos_defense_theta = np.clip(cos_defense_theta, -1.0, 1.0)
                defense_theta = np.arccos(cos_defense_theta)
                theta = min(theta, defense_theta)

        # The closer theta is to 0, the faster the reaction
        self.reaction_time = 0.1 + 1.0 * theta / np.pi

    def calc_time_to_intercept(self, target_position: np.ndarray, param: dict) -> float:
        self.UPPCF = 0.0  # initialise this for later
        self.wUPPCF = 0.0  # initialise this for later

        distance = np.linalg.norm(target_position - self.position)

        # Handle zero distance case
        if distance < 1e-10:
            self.time_to_intercept = 0.0
            return self.time_to_intercept

        direction = (target_position - self.position) / distance

        speed_towards_target = np.dot(self.velocity, direction)

        acceleration_time = max(
            (param["max_player_speed"] - speed_towards_target)
            / param["max_player_accel"],
            0,
        )

        acceleration_distance = (
            speed_towards_target * acceleration_time
            + 0.5 * param["max_player_accel"] * acceleration_time**2
        )

        if acceleration_distance >= distance:
            acceleration_time_needed = (
                -speed_towards_target
                + math.sqrt(
                    speed_towards_target**2 + 2 * param["max_player_accel"] * distance
                )
            ) / param["max_player_accel"]
            self.time_to_intercept = acceleration_time_needed
            return self.time_to_intercept

        remain_distance = distance - acceleration_distance
        constant_speed_time = remain_distance / param["max_player_speed"]

        self.time_to_intercept = acceleration_time + constant_speed_time
        return self.time_to_intercept

    def probability_intercept_ball(self, T: float) -> float:
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1 / (
            1.0
            + np.exp(
                -np.pi / np.sqrt(3.0) / self.tti_sigma * (T - self.time_to_intercept)
            )
        )
        return f


""" Generate pitch control map """


def generate_pitch_control_for_event(
    event_id: int,
    events: pd.DataFrame,
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    removed_players: pd.DataFrame,
    params: dict,
    field_dimen: Tuple[float, float] = (94.0, 37.0),
    n_grid_cells_x: int = 50,
    remove: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[int], list]:
    """generate_pitch_control_for_event

    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)

    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        removed_players: Dataframe containing the players to be removed from the pitch control calculation for each frame
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (94.0,37.0)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        remove: Boolean flag to indicate whether to remove players near the disc from the pitch control calculation.

    Returns
    -----------
        UPPCFa / (UPPCFa + UPPCFd): Normalized pitch control surface for the attacking team (dimen (n_grid_cells_x,n_grid_cells_y) )
               Surface for the defending team is just 1-UPPCFa.
        frame_player_UPPCF: Pitch control contribution for each attacking player at each location on the pitch (dimen (num_attacking_players,n_grid_cells_y,n_grid_cells_x) )
        defending_removed_player: Player id of the defending player removed from the calculation (if any)
        attacking_players: List of attacking player objects used in the calculation
    """
    # get the details of the event (frame, team in possession, ball_start_position)
    frame = events.loc[event_id]["Start Frame"]
    ball_start_pos = np.array(
        [events.loc[event_id]["Start X"], events.loc[event_id]["Start Y"]]
    )
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])

    # Use n_grid_cells_x and n_grid_cells_y directly as the number of cells
    xgrid = np.linspace(-field_dimen[0] / 2.0, field_dimen[0] / 2.0, n_grid_cells_x)
    ygrid = np.linspace(-field_dimen[1] / 2.0, field_dimen[1] / 2.0, n_grid_cells_y)

    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    # initialise pitch control grids for attacking and defending teams
    UPPCFa = np.zeros(shape=(len(ygrid), len(xgrid)))
    UPPCFd = np.zeros(shape=(len(ygrid), len(xgrid)))
    # if remove is true, remove players near the disc
    if remove is False:
        attacking_players, defending_players, _ = initialise_players(
            tracking_home.loc[frame - 1],
            tracking_away.loc[frame - 1],
            "Home",
            "Away",
            params,
            None,
        )
    else:
        attacking_players, defending_players, defending_removed_player = (
            initialise_players(
                tracking_home.loc[frame - 1],
                tracking_away.loc[frame - 1],
                "Home",
                "Away",
                params,
                removed_players.loc[removed_players["Frame"] == frame, "Player"],
            )
        )
    frame_player_UPPCF = np.zeros(
        shape=(len(attacking_players), len(ygrid), len(xgrid))
    )
    # calculate pitch pitch control model at each location on the pitch
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            UPPCFa[i, j], UPPCFd[i, j], frame_player_UPPCF[:, i, j] = (
                calculate_pitch_control_at_target(
                    target_position,
                    attacking_players,
                    defending_players,
                    ball_start_pos,
                    params,
                )
            )

    return (
        UPPCFa / (UPPCFa + UPPCFd),
        frame_player_UPPCF,
        defending_removed_player,
        attacking_players,
    )


def calculate_pitch_control_at_target(
    target_position: np.ndarray,
    attacking_players: list,
    defending_players: list,
    ball_start_pos: np.ndarray,
    params: dict,
) -> Tuple[float, float, np.ndarray]:
    """calculate_pitch_control_at_target

    Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.

    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )

    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
        grid_player_UPPCF: Pitch control contribution for each attacking player at the target position (dimen (num_attacking_players,) )

    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(
        np.isnan(ball_start_pos)
    ):  # assume that ball is already at location
        ball_travel_time = 0.0
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = (
            np.linalg.norm(target_position - ball_start_pos)
            / params["average_ball_speed"]
        )

    # Sort attacking players by id
    attacking_players = sorted(attacking_players, key=lambda x: int(x.id))

    # Calculate time to intercept for all players first
    for p in attacking_players:
        p.calc_time_to_intercept(target_position, params)
    for p in defending_players:
        p.calc_time_to_intercept(target_position, params)

    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin([p.time_to_intercept for p in attacking_players])
    tau_min_def = np.nanmin([p.time_to_intercept for p in defending_players])

    # solve pitch control model by integrating equation 3 in Spearman et al.
    grid_player_UPPCF = np.zeros(shape=(len(attacking_players), 1))
    # first remove any player that is far (in time) from the target location
    attacking_players = [
        p
        for p in attacking_players
        if p.time_to_intercept - tau_min_att < params["time_to_control_att"]
    ]
    defending_players = [
        p
        for p in defending_players
        if p.time_to_intercept - tau_min_def < params["time_to_control_def"]
    ]
    # set up integration arrays
    dT_array = np.arange(
        ball_travel_time - params["int_dt"],
        ball_travel_time + params["max_int_time"],
        params["int_dt"],
    )
    UPPCFatt = np.zeros_like(dT_array)
    UPPCFdef = np.zeros_like(dT_array)
    # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
    ptot = 0.0
    i = 1
    while 1 - ptot > params["model_converge_tol"] and i < dT_array.size:
        T = dT_array[i]
        for player in attacking_players:
            # calculate ball control probablity for 'player' in time interval T+dt
            dUPPCFdT = (
                (1 - UPPCFatt[i - 1] - UPPCFdef[i - 1])
                * player.probability_intercept_ball(T)
                * player.lambda_att
            )
            # make sure it's greater than zero
            assert (
                dUPPCFdT >= 0
            ), "Invalid attacking player probability (calculate_pitch_control_at_target)"
            player.UPPCF += (
                dUPPCFdT * params["int_dt"]
            )  # total contribution from individual player
            UPPCFatt[
                i
            ] += (
                player.UPPCF
            )  # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
        for player in defending_players:
            # calculate ball control probablity for 'player' in time interval T+dt
            dUPPCFdT = (
                (1 - UPPCFatt[i - 1] - UPPCFdef[i - 1])
                * player.probability_intercept_ball(T)
                * player.lambda_def
            )
            # make sure it's greater than zero
            assert (
                dUPPCFdT >= 0
            ), "Invalid defending player probability (calculate_pitch_control_at_target)"
            player.UPPCF += (
                dUPPCFdT * params["int_dt"]
            )  # total contribution from individual player
            UPPCFdef[i] += player.UPPCF  # add to sum over players in the defending team
        ptot = UPPCFdef[i] + UPPCFatt[i]  # total pitch control probability
        i += 1
    for n, player in enumerate(attacking_players):
        grid_player_UPPCF[n] = player.UPPCF

    return UPPCFatt[i - 1], UPPCFdef[i - 1], grid_player_UPPCF[:, 0]


def calculate_ultimate_pitch_control(
    UPPCF: np.ndarray,
    player_UPPCF: np.ndarray,
    ball_start_pos: np.ndarray,
    stalling_pos: np.ndarray,
    field_dimen=(94, 37),
) -> Tuple[np.ndarray, np.ndarray]:
    """calculate_ultimate_pitch_control
    Adjusts the standard pitch control surface to account for the unique aspects of ultimate frisbee, namely the disc's range and stalling rules.

    Parameters
    -----------
        UPPCF: Standard pitch control surface (dimen (time, n_grid_cells_y, n_grid_cells_x) )
        player_UPPCF: Pitch control contribution for each attacking player at each location on the pitch (dimen (num_attacking_players, time, n_grid_cells_y, n_grid_cells_x) )
        ball_start_pos: Current position of the disc (start position for a pass) for each time frame (dimen (time, 2) )
        stalling_pos: Position of the stalling marker for each time frame (dimen (time, 2) )
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (94.0,37.0)

    Returns
    -----------
        wUPPCF: Adjusted pitch control surface for ultimate frisbee (dimen (time, n_grid_cells_y, n_grid_cells_x) )
        player_wUPPCF: Adjusted pitch control contribution for each attacking player at each location on the pitch (dimen (num_attacking_players, time, n_grid_cells_y, n_grid_cells_x) )
    """

    def sigmoid(a, b, x):
        s = 1 / (1 + np.exp(-(a * (x - b))))
        return s

    def line_intersection(p1, p2, p3, p4):
        """
        Find the intersection point of two lines (p1, p2) and (p3, p4).
        Returns the intersection point if it exists, otherwise returns None.
        """
        s1_x = p2[0] - p1[0]
        s1_y = p2[1] - p1[1]
        s2_x = p4[0] - p3[0]
        s2_y = p4[1] - p3[1]

        denom = -s2_x * s1_y + s1_x * s2_y
        if np.isclose(denom, 0):
            return None  # Lines are parallel

        s = (-s1_y * (p1[0] - p3[0]) + s1_x * (p1[1] - p3[1])) / denom
        t = (s2_x * (p1[1] - p3[1]) - s2_y * (p1[0] - p3[0])) / denom

        if 0 <= s <= 1 and 0 <= t <= 1:
            # Collision detected
            i_x = p1[0] + (t * s1_x)
            i_y = p1[1] + (t * s1_y)
            return np.array([i_x, i_y])
        return None

    weight_range = np.zeros_like(UPPCF)
    weight_stalling = np.zeros_like(UPPCF)
    dx = field_dimen[0] / weight_range.shape[2]
    dy = field_dimen[1] / weight_range.shape[1]
    for t in range(weight_range.shape[0]):
        continue_value = False
        if np.linalg.norm(ball_start_pos[t] - stalling_pos[t]) > 3:
            weight_stalling[t] = np.ones(
                (weight_stalling.shape[1], weight_stalling.shape[2])
            )
            continue_value = True
        for i in range(weight_range.shape[1]):
            for j in range(weight_range.shape[2]):
                # Calculate weight for range
                target_x = j * dx + dx / 2
                target_y = i * dy + dy / 2
                target = np.array([target_x, target_y])
                dist = np.linalg.norm(target - ball_start_pos[t])
                y1 = (1 - sigmoid(0.15, 40, dist) + 1) / 2
                y2 = (1 - sigmoid(0.07, 40, dist) + 1) / 2
                y3 = 1 - sigmoid(2, 70, dist)
                y = (((y1 + y2) / 2) + y3) / 2
                weight_range[t, i, j] = y

                if continue_value:
                    continue

                # Calculate weight for stalling
                disc_to_target = target - ball_start_pos[t]
                disc_to_stalling = stalling_pos[t] - ball_start_pos[t]

                dist_to_target = np.linalg.norm(disc_to_target)
                r = 1 - min((dist_to_target / 30), 1)

                # Calculate angle of the disc-to-stalling vector
                angle = np.angle(disc_to_stalling[0] + disc_to_stalling[1] * 1j)
                stalling_one_hand = np.array(
                    [
                        stalling_pos[t][0] + np.sin(angle) * r,
                        stalling_pos[t][1] - np.cos(angle) * r,
                    ]
                )
                stalling_other_hand = np.array(
                    [
                        stalling_pos[t][0] - np.sin(angle) * r,
                        stalling_pos[t][1] + np.cos(angle) * r,
                    ]
                )
                # Find the intersection of stalling_one_hand_to_stalling_other_hand and disc_to_target
                dist_to_target = np.linalg.norm(disc_to_target)
                intersection = line_intersection(
                    stalling_one_hand, stalling_other_hand, ball_start_pos[t], target
                )
                if intersection is not None:
                    stalling_to_intersection = np.linalg.norm(
                        intersection - stalling_pos[t]
                    )
                    stalling_to_hand = np.linalg.norm(
                        stalling_one_hand - stalling_pos[t]
                    )
                    weight_stalling[t, i, j] = (
                        sigmoid(12, 0.5, stalling_to_intersection / stalling_to_hand)
                        + 1
                    ) / 2
                else:
                    weight_stalling[t, i, j] = 1
        weight_stalling[t] = weight_stalling[t][::-1, :]

    wUPPCF = UPPCF * weight_range * weight_stalling
    player_wUPPCF = (
        player_UPPCF * weight_range[None, :, :, :] * weight_stalling[None, :, :, :]
    )

    return wUPPCF, player_wUPPCF
