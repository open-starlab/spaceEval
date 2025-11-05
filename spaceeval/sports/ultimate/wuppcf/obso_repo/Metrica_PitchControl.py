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

import numpy as np
import pandas as pd


def initialise_players(
    attacking_team,
    defending_team,
    attacking_teamname,
    defending_teamname,
    params,
    removed_player,
    field_dimen=(47.0, 18.5),
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
    attacking_player_ids = np.arange(0, 7)
    defending_player_ids = np.arange(0, 7)

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
                if _[5] == str(int(removed_player)) and len(_) == 8
            ]
        )
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
    def __init__(self, pid, team, opponent_team, teamname, params):
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

    def get_position(self, team):
        # MultiIndex対応: team.indexがMultiIndexの場合とSeriesの場合で処理を分ける
        if isinstance(team.index, pd.MultiIndex):
            # MultiIndex: (teamname, pid, column_name) の形式
            # 該当するプレイヤーの列を検索
            x_col = None
            y_col = None
            for idx in team.index:
                if idx[0] == self.teamname and idx[1] == str(self.id):
                    if x_col is None:
                        x_col = idx  # 最初の列がx座標
                    elif y_col is None:
                        y_col = idx  # 2番目の列がy座標
                        break

            if x_col is not None and y_col is not None:
                self.position = np.array([team[x_col], team[y_col]])
            else:
                self.position = np.array([np.nan, np.nan])
        else:
            self.position = np.array(
                [team[self.playername + "x"], team[self.playername + "y"]]
            )
        self.inframe = not np.any(np.isnan(self.position))

    def get_velocity(self, team):
        # MultiIndex対応
        if isinstance(team.index, pd.MultiIndex):
            vx_col = None
            vy_col = None
            for idx in team.index:
                if idx[0] == self.teamname and idx[1] == str(self.id):
                    if "_vx" in str(idx[2]):
                        vx_col = idx
                    elif "_vy" in str(idx[2]):
                        vy_col = idx

            if vx_col is not None and vy_col is not None:
                self.velocity = np.array([team[vx_col], team[vy_col]])
            else:
                self.velocity = np.array([0.0, 0.0])
        else:
            self.velocity = np.array(
                [team[self.playername + "vx"], team[self.playername + "vy"]]
            )

        if np.any(np.isnan(self.velocity)):
            self.velocity = np.array([0.0, 0.0])

    def get_reaction_time(self, team, opponent_team):
        disc_pos = np.array([team["disc_x"], team["disc_y"]])
        direction_to_disc = disc_pos - self.position
        direction_velocity = self.velocity
        # angle from direction_to_disc to direction_velocity
        theta = np.arccos(
            np.dot(direction_to_disc, direction_velocity)
            / (np.linalg.norm(direction_to_disc) * np.linalg.norm(direction_velocity))
        )
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
            # angle from direction_to_disc to direction_to_offense
            defense_theta = np.arccos(
                np.dot(direction_to_disc, direction_to_offense)
                / (
                    np.linalg.norm(direction_to_disc)
                    * np.linalg.norm(direction_to_offense)
                )
            )
            theta = min(theta, defense_theta)
        # The closer theta is to 0, the faster the reaction
        self.reaction_time = 0.1 + 1.0 * theta / np.pi

    # def simple_time_to_intercept(self, r_final):
    #     self.UPPCF = 0.0  # initialise this for later
    #     self.wUPPCF = 0  # initialise this for later
    #     # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
    #     # and then runs at full speed to the target position.
    #     r_reaction = self.position + self.velocity * self.reaction_time
    #     self.time_to_intercept = (
    #         self.reaction_time + np.linalg.norm(r_final - r_reaction) / self.vmax
    #     )
    #     return self.time_to_intercept

    def calc_time_to_intercept(self, r_final, param):
        distance = np.linalg.norm(r_final - self.position)
        direction = (r_final - self.position) / distance

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
            return acceleration_time_needed

        remain_distance = distance - acceleration_distance
        constant_speed_time = remain_distance / param["max_player_speed"]

        self.time_to_intercept = acceleration_time + constant_speed_time
        np.set_printoptions(precision=2)
        # print(f'from:{self.position}, to:{r_final}, speed:{self.velocity}, time:{self.time_to_intercept}')
        return self.time_to_intercept

    def probability_intercept_ball(self, T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1 / (
            1.0
            + np.exp(
                -np.pi / np.sqrt(3.0) / self.tti_sigma * (T - self.time_to_intercept)
            )
        )
        return f


""" Generate pitch control map """


def default_model_params(time_to_control_veto=3):
    """
    default_model_params()

    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.

    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.


    Returns
    -----------

    params: dictionary of parameters required to determine and calculate the model

    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params["max_player_accel"] = (
        6.2 / 2
    )  # maximum player acceleration m/s/s, not used in this implementation
    params["max_player_speed"] = 6.9 / 2  # maximum player speed m/s
    params["reaction_time"] = (
        0.1  # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    )
    params["tti_sigma"] = (
        0.45  # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    )
    params["kappa_def"] = (
        1.2  # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    )
    params["lambda_att"] = 4.3  # ball control parameter for attacking team
    params["lambda_def"] = (
        4.3  # * params['kappa_def'] # ball control parameter for defending team
    )
    params["average_ball_speed"] = 15.44 / 2  # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params["int_dt"] = 1 / 15  # integration timestep (dt)
    params["max_int_time"] = 5  # upper limit on integral time
    params["model_converge_tol"] = (
        0.01  # assume convergence when PPCF>0.99 at a given location.
    )
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start.
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params["time_to_control_att"] = (
        time_to_control_veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_def"])
    )
    return params


def generate_pitch_control_for_event(
    event_id,
    events,
    tracking_home,
    tracking_away,
    removed_players,
    params,
    field_dimen=(47.0, 18.5),
    n_grid_cells_x=50,
    remove=False,
):
    """generate_pitch_control_for_event

    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)

    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.

    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')

    Returrns
    -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """
    # get the details of the event (frame, team in possession, ball_start_position)
    frame = events.loc[event_id]["Start Frame"]
    ball_start_pos = np.array(
        [events.loc[event_id]["Start X"], events.loc[event_id]["Start Y"]]
    )
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = (
        np.linspace(0, n_grid_cells_x, int(np.ceil(field_dimen[0]))) * dx
        - field_dimen[0] / 2.0
        + dx / 2.0
    )
    ygrid = (
        np.linspace(0, n_grid_cells_y, int(np.ceil(field_dimen[1]))) * dy
        - field_dimen[1] / 2.0
        + dy / 2.0
    )
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    # initialise pitch control grids for attacking and defending teams
    UPPCFa = np.zeros(shape=(len(ygrid), len(xgrid)))
    UPPCFd = np.zeros(shape=(len(ygrid), len(xgrid)))
    # if remove is true, remove players near the disc
    if remove is False:
        attacking_players, defending_players, _ = initialise_players(
            tracking_home.loc[frame],
            tracking_away.loc[frame],
            "Home",
            "Away",
            params,
            None,
        )
    else:
        attacking_players, defending_players, defending_removed_player = (
            initialise_players(
                tracking_home.loc[frame],
                tracking_away.loc[frame],
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
        defending_players,
    )


def calculate_pitch_control_at_target(
    target_position, attacking_players, defending_players, ball_start_pos, params
):
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

    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin(
        [p.calc_time_to_intercept(target_position, params) for p in attacking_players]
    )
    tau_min_def = np.nanmin(
        [p.calc_time_to_intercept(target_position, params) for p in defending_players]
    )

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
    # if i>=dT_array.size:
    # print("Integration failed to converge: %1.3f" % (ptot) )
    for n, player in enumerate(attacking_players):
        grid_player_UPPCF[n] = player.UPPCF

    return UPPCFatt[i - 1], UPPCFdef[i - 1], grid_player_UPPCF[:, 0]


def calculate_ultimate_pitch_control(
    UPPCF,
    player_UPPCF,
    ball_start_pos,
    stalling_pos,
    attacking_players,
    defending_players,
    field_dimen=(94, 37),
):
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
                # stalling_one_hand_to_stalling_other_handとdisc_to_targetの交点を求める。互いの線分上に交点がない場合、交点はなしとする
                # 交点がある場合、stalling_pos[t]と交点の距離をrとする
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
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # im = plt.imshow(weight_range[52], cmap='Blues', interpolation='spline36', vmin=0.7, vmax=1)
    # plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    # cbar = plt.colorbar(im)
    # cbar.ax.set_aspect(50)  # Adjust the aspect ratio to match the figure height
    # plt.savefig('weight_range.png')
    # plt.close()
    # plt.figure(figsize=(10, 5))
    # im = plt.imshow(weight_stalling[52], cmap='Blues', interpolation='spline36', vmin=0.5, vmax=1)
    # plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    # cbar = plt.colorbar(im)
    # cbar.ax.set_aspect(50)  # Adjust the aspect ratio to match the figure height
    # plt.savefig('weight_stalling.png')
    # plt.close()
    return wUPPCF, player_wUPPCF
    return wUPPCF, player_wUPPCF
