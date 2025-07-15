import numpy as np
import pandas as pd
import math
import importlib.resources as pkg_resources
import yaml

from ..utils import get_residual_param as res
from ...basketball import utils
from ..utils import SportVU_IO as sio
from  ...basketball import model_parameter

rate_pass_array = [
    0.08303125, 0.25315865924849945, 0.5075870844599968, 0.6399647600091363, 
    0.6770451991735836, 0.6504492120379007, 0.6191329005298112, 0.5732106091800896
    ]
rate_dribble_array = [
    0.91696875, 0.7468413407515005, 0.49241291554000316, 0.36003523999086373, 
    0.3229548008264163, 0.34955078796209926, 0.3808670994701888, 0.42678939081991035
    ]

# Pass and dribble velocity [m/s] array as a function of travel distance
pass_velocity_array = [
    3.7775490360362016, 6.304588948842771, 7.300270466644174, 8.199174281521207, 9.124375876694469, 
    9.839056331861283, 10.369172485611765, 10.642227440584952, 10.807464781665454, 10.819215769367302
    ]
dribble_velocity_array = [
    2.2436922778753385, 2.615258991319709, 2.763648245714895, 2.8703517662206273, 3.033426275964569, 
    3.1861691897985542, 3.3521223007518937, 3.473300253216001, 3.5019421441137153, 3.4846248973174854
    ]
    
class BMOS():
    def __init__(self,data):

        with pkg_resources.open_text(model_parameter,'params.yaml') as f:
            params = yaml.safe_load(f)

        self.accel = float(params["accel"])
        self.kappa = float(params["kappa"])
        self.lam = float(params["lam"])
        self.att_reaction_time = float(params["att_reaction_time"])
        self.def_reaction_time = float(params["def_reaction_time"])
        self.integral_xmin = float(params["integral_xmin"])

        self.params = default_model_params(self.accel, self.kappa, self.lam, self.att_reaction_time, self.def_reaction_time)
    
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Invalid data format: must be a non-empty pandas DataFrame")
    
        # Vérifiez que les colonnes nécessaires existent
        required_columns = ['x_ball', 'y_ball', 'ball_holder_pid_idx']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        with pkg_resources.open_text(utils,'scoremodel_basket.csv') as f:
            score = np.array(pd.read_csv(f,header=None))
        score = score/np.max(score)

        self.fit_params = [float(params["player_accel"]), float(params["att_reaction_time"]), float(params["player_max_speed_att"])]

        PPCFa = generate_pitch_control_for_event(data, self.params, self.fit_params, self.integral_xmin)

        transition_plot = sio.make_transitionmodel_for_event(data)

        self.values = PPCFa * score * transition_plot

    def get_values(self):
        return self.values
    
    
class Player:
    def __init__(self, pid_idx, data, teamside, params, fit_params, integral_xmin):
        self.id_idx = pid_idx                            # player index within 10 players, return 0-9
        self.bid = self.get_player_id_with_ball(data)  # player index (0-9) with ball, return None if the ball isn't holded
        self.teamside = teamside                         # 'attacker' or 'defender'
        self.params = params                             # parameter dictionary
        self.position = self.get_position(data)        # player position np.array([x, y])
        self.velocity = self.get_velocity(data)        # player velocity np.array([vx, vy])
        self.fit_params = fit_params                     # fitted parameter for tau_true - tau_exp distribution
        self.integral_xmin = integral_xmin               # xmin for tau_true - tau_exp distribution integration 
        self.time_to_intercept = 0.0
        self.probability_to_intercept = 0.0
        self.PPCF = 0.0
        
    def get_values(self):
        return self.values
    
    def get_player_id_with_ball(self, data):
        if data['ball_holder_pid_idx'].values[0] == 0:
            return None
        else:
            return int(data['ball_holder_pid_idx'].values[0]) - 1

    def get_position(self, data):
        if self.teamside == 'attacker':
            pos = np.array([data[f'x_att{self.id_idx}'].values[0], data[f'y_att{self.id_idx}'].values[0]])
            return pos if not np.any(np.isnan(pos)) else np.array([0.0, 0.0])
        else:
            pos = np.array([data[f'x_def{self.id_idx}'].values[0], data[f'y_def{self.id_idx}'].values[0]])
            return pos if not np.any(np.isnan(pos)) else np.array([0.0, 0.0])

    def get_velocity(self, data):
        if self.teamside == 'attacker':
            pos = np.array([data[f'vx_att{self.id_idx}'].values[0], data[f'vy_att{self.id_idx}'].values[0]])
            return pos if not np.any(np.isnan(pos)) else np.array([0.0, 0.0])
        else:
            pos = np.array([data[f'vx_def{self.id_idx}'].values[0], data[f'vy_def{self.id_idx}'].values[0]])
            return pos if not np.any(np.isnan(pos)) else np.array([0.0, 0.0])

    def reset_PPCF(self):
        self.PPCF = 0

    def simple_time_to_intercept(self, r_final):
        """
        Return time taken for a player to reach target position r_final.
        It assumes the player moves at constant acceleration accel with realistic maximum velocity vmax.
        """
        accel, vini = self.params['player_accel'], np.linalg.norm(self.velocity)
        if self.teamside == 'attacker':
            vmax = self.params['player_max_speed_att']
            if self.id_idx == self.bid:
                r_time = self.params['possesor_reaction_time']
            else:
                r_time = self.params['att_reaction_time']
        else:
            vmax = self.params['player_max_speed_def']
            r_time = self.params['def_reaction_time']
    
        # calc position after reaction time
        adjust_position = self.position + self.velocity * r_time

        # calc time to reach target position from adjust_position, given vmax isn't set
        t = (-vini / accel + np.sqrt(vini ** 2 / accel ** 2 + 2 * np.linalg.norm(r_final - adjust_position) / accel))
        
        # consider the situation velocity exceeds vmax
        if vini + t * accel > vmax:
            limit_time = (vmax - vini) / accel
            remaining_distance = np.linalg.norm(r_final - adjust_position) - (vini * limit_time + 0.5 * accel * limit_time ** 2) 
            self.time_to_intercept = r_time + limit_time + remaining_distance / vmax
        else:
            self.time_to_intercept = r_time + t

        return self.time_to_intercept
    
    def probability_intercept_ball(self, T):
        """
        Return propability that a player reaches target position before the ball, T,
        by integrating tau_true - tau_exp distribution.
        """
        if self.fit_params[0] == 0:
            # when fitting does not go well
            self.probability_to_intercept = 0
        else:
            # integrate from integral_xmin to T-time_to_intercept
            p_int = res.get_cdf_value(T - self.time_to_intercept, self.fit_params, self.integral_xmin)
            self.probability_to_intercept = p_int
            
        return self.probability_to_intercept

def initialise_players(data, teamside, params, fit_params, integral_xmin):
    player_ids_idx = [i for i in range(5)]
    return [Player(pid_idx, data, teamside, params, fit_params, integral_xmin) for pid_idx in player_ids_idx]

def default_model_params(accel, kappa, lam, att_reaction_time, def_reaction_time):
    params = {
        'player_accel': accel,                  # [m/s^2]
        'player_max_speed_att': 5.00,           # [m/s]
        'player_max_speed_def': 5.00,           # [m/s]
        'att_reaction_time': att_reaction_time, # [s]
        'possesor_reaction_time': 0.,           # [s]
        'def_reaction_time': def_reaction_time, # [s]

        'lambda_att': lam,
        'lambda_att_bid': lam,
        'lambda_def': kappa * lam,

        'int_dt': 0.01,
        'max_int_time': 10.,
        'model_converge_tol': 0.01,
        'probability_to_control': 0,
    }
    return params

def process_relevant_players(revevant_players, players, tau_min):
    """
    Return a closest player when array revevant_players does not include any players.
    No process happens when revevant_players contains any players.
    """
    if len(revevant_players) == 0:
        revevant_players = [p for p in players if p.time_to_intercept==tau_min]
    return revevant_players

def calculate_ball_travel_time(ball_start_pos, target_position, ball_speed):
    """
    Return ball travel time from ball_start_pos to target_position with constant ball_speed.
    """
    if ball_start_pos is None or np.isnan(ball_start_pos).any():
        ball_travel_time = 0.0
    else:
        ball_travel_time = np.linalg.norm(target_position - ball_start_pos) / ball_speed

    assert ball_travel_time >= 0, "Ball travel time is less than zero"
    return ball_travel_time

def generate_pitch_control_for_event(data, params, fit_params, integral_xmin, field_dimen=(28., 15.), n_grid_cells_x=28):
    """
    Return PPCF/PBCF 2d array and relevant players information. Variables explanation:
    t_data: tracking dataset per game
    s_id: scene id
    f_id: frame id
    params: default_model_params()
    fit_parames, integration_xmin: parameters used for probability_intercept_ball()
    """

    # current ball position (x, y)
    ball_start_pos = np.array([data['x_ball'].values[0], data['y_ball'].values[0]])

    # set grid information
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx, dy = field_dimen[0] / n_grid_cells_x, field_dimen[1] / n_grid_cells_y
    xgrid, ygrid = np.linspace(dx / 2, field_dimen[0] - dx / 2, n_grid_cells_x), np.linspace(dy / 2, field_dimen[1] - dy / 2, n_grid_cells_y)

    # set resulting array
    PPCFa, PPCFd = np.zeros((n_grid_cells_y, n_grid_cells_x)), np.zeros((n_grid_cells_y, n_grid_cells_x))
    rel_att_ids, rel_def_ids = np.empty((n_grid_cells_y, n_grid_cells_x), dtype=object), np.empty((n_grid_cells_y, n_grid_cells_x), dtype=object)

    # initialize players information
    attacking_players, defending_players = initialise_players(data, 'attacker', params, fit_params, integral_xmin), \
                                            initialise_players(data, 'defender', params, fit_params, integral_xmin)

    for i, y in enumerate(ygrid):
        for j, x in enumerate(xgrid):

            # calc distance idx from current ball position to target position
            idx_distance = math.floor(np.linalg.norm(np.array([x, y]) - ball_start_pos))

            # get ball pass and dribble velocity and rate depending on distance
            if 0 <= idx_distance <= 7:
                rate_pass = rate_pass_array[idx_distance]
                rate_dribble = rate_dribble_array[idx_distance]
            else:
                rate_pass, rate_dribble = 0.5, 0.5
            
            if 0 <= idx_distance <= 9:
                bv_pass = pass_velocity_array[idx_distance]
                bv_dribble = dribble_velocity_array[idx_distance]
            else:
                bv_pass = pass_velocity_array[-1]
                bv_dribble = dribble_velocity_array[-1]

            rel_att_ids_pass, rel_def_ids_pass, PPCFa_pass, PPCFd_pass = calculate_ppcf_pass(
                                                                                    np.array([x, y]), attacking_players, defending_players, 
                                                                                    ball_start_pos, params, bv_pass)
            _, _, PPCFa_dribble, PPCFd_dribble = calculate_ppcf_dribble(
                                                        np.array([x, y]), attacking_players, defending_players, 
                                                        ball_start_pos, params, bv_dribble)
            
            rel_att_ids[i, j], rel_def_ids[i, j] = rel_att_ids_pass, rel_def_ids_pass        
            PPCFa[i, j], PPCFd[i, j] = rate_pass * PPCFa_pass + rate_dribble * PPCFa_dribble, rate_pass * PPCFd_pass + rate_dribble * PPCFd_dribble

    return np.flipud(PPCFa)

def calculate_ppcf_pass(target_position, attacking_players, defending_players, ball_start_pos, params, ball_velocity):
   
    get_time_to_intercept = lambda players: np.nanmin([p.simple_time_to_intercept(target_position) for p in players if p.id_idx != p.bid])
    filter_players = lambda players: [p for p in players 
                                        if (p.probability_intercept_ball(ball_travel_time) > params['probability_to_control']) 
                                        and p.id_idx != p.bid]
    for player in attacking_players:
        player.reset_PPCF()
    for player in defending_players:
        player.reset_PPCF()

    ball_travel_time = calculate_ball_travel_time(ball_start_pos,target_position, ball_velocity)
    tau_min_att, tau_min_def = get_time_to_intercept(attacking_players), get_time_to_intercept(defending_players)
    relevant_attackers, relevant_defenders = filter_players(attacking_players), filter_players(defending_players)
    
    relevant_attackers, relevant_defenders = process_relevant_players(relevant_attackers, attacking_players, tau_min_att), \
                                                process_relevant_players(relevant_defenders, defending_players, tau_min_def)

    rel_att_id, rel_def_id = [p.id_idx for p in relevant_attackers], [p.id_idx for p in relevant_defenders]

    dt_array = np.arange(ball_travel_time - params['int_dt'], ball_travel_time + params['max_int_time'], params['int_dt'])
    PPCFatt, PPCFdef = np.zeros_like(dt_array), np.zeros_like(dt_array)

    i = 1
    while PPCFatt[i-1] + PPCFdef[i-1] < 1 - params['model_converge_tol'] and i < dt_array.size:
        for player in relevant_attackers:
            dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_to_intercept * player.params['lambda_att']
            player.PPCF += dPPCFdT * params['int_dt']
            PPCFatt[i] += player.PPCF

        for player in relevant_defenders:
            dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_to_intercept * player.params['lambda_def']
            player.PPCF += dPPCFdT * params['int_dt']
            PPCFdef[i] += player.PPCF
        i += 1

    if i < 3:
        return rel_att_id, rel_def_id, PPCFatt[i-1] / (PPCFatt[i-1] + PPCFdef[i-1]), PPCFdef[i-1] / (PPCFatt[i-1] + PPCFdef[i-1])
    else:
        if PPCFatt[i-1] + PPCFdef[i-1] < 1 - params['model_converge_tol'] and i >= dt_array.size:
            return rel_att_id, rel_def_id, PPCFatt[i-1], PPCFdef[i-1]
        else:
            return  rel_att_id, rel_def_id, PPCFatt[i-2], PPCFdef[i-2]

def calculate_ppcf_dribble(target_position, attacking_players, defending_players, ball_start_pos, params, ball_velocity):
    get_time_to_intercept = lambda players: np.nanmin([p.simple_time_to_intercept(target_position) for p in players])
    filter_players = lambda players: [p for p in players if (p.probability_intercept_ball(ball_travel_time) > params['probability_to_control'])]

    # make attackers relevant player array which only includes ball possessor
    relevant_attackers = [p for p in attacking_players if p.id_idx == p.bid]

    if not relevant_attackers:
        return [0], [0], 0, 0
    else:
        for player in attacking_players:
            player.reset_PPCF()
        for player in defending_players:
            player.reset_PPCF()

        ball_travel_time = calculate_ball_travel_time(ball_start_pos, target_position, ball_velocity)
        tau_min_def = get_time_to_intercept(defending_players)
        relevant_defenders = filter_players(defending_players)
        
        relevant_defenders = process_relevant_players(relevant_defenders, defending_players, tau_min_def)

        rel_att_id, rel_def_id = [p.id_idx for p in relevant_attackers], [p.id_idx for p in relevant_defenders]

        dt_array = np.arange(ball_travel_time - params['int_dt'], ball_travel_time + params['max_int_time'], params['int_dt'])
        PPCFatt, PPCFdef = np.zeros_like(dt_array), np.zeros_like(dt_array)

        i = 1
        while PPCFatt[i-1] + PPCFdef[i-1] < 1 - params['model_converge_tol'] and i < dt_array.size:
            for player in relevant_attackers:
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_intercept_ball(ball_travel_time) * player.params['lambda_att_bid']
                player.PPCF += dPPCFdT * params['int_dt']
                PPCFatt[i] += player.PPCF

            for player in relevant_defenders:
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_to_intercept * player.params['lambda_def']
                player.PPCF += dPPCFdT * params['int_dt']
                PPCFdef[i] += player.PPCF
            i += 1

        if i < 3:
            return rel_att_id, rel_def_id, PPCFatt[i-1] / (PPCFatt[i-1] + PPCFdef[i-1]), PPCFdef[i-1] / (PPCFatt[i-1] + PPCFdef[i-1])
        else:
            if PPCFatt[i-1] + PPCFdef[i-1] < 1 - params['model_converge_tol'] and i >= dt_array.size:
                return rel_att_id, rel_def_id, PPCFatt[i-1], PPCFdef[i-1]
            else:
                return  rel_att_id, rel_def_id, PPCFatt[i-2], PPCFdef[i-2]

