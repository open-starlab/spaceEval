import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import skewcauchy
from scipy.optimize import curve_fit
from . import SportVU_IO as sio

# event data index
EVENT_LABEL = 1
BALL_PID_IDX = 5
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

# tracking data (t_data) index
PLAYER_POSITIONS = slice(0, 20)
PLAYER_VELOCITIES = slice(23, 43)
SHOT_CLOCK_ID = 47

# Constant
N_TEST_GAMES = 50 # use the first 50 games to shorten the processing time

def get_params(accel, r_time, vmax):
    hist_array = []

    # obtain tau_true - tau_exp array as hist_array
    for game in tqdm(np.arange(1, N_TEST_GAMES+1)):
        # load trackiing data and on-ball event data
        t_data = sio.load_tracking_data(game, onball=True)
        e_data = sio.load_event_data(game, onball=True)

        for t_data_scene, e_data_scene in zip(t_data, e_data):
            if not len(e_data_scene) >= 2:
                continue

            for t_data_frame, t_data_frame_next, e_data_frame, e_data_frame_next \
                                in zip(t_data_scene[:-1], t_data_scene[1:], e_data_scene[:-1], e_data_scene[1:]):
                if e_data_frame[EVENT_LABEL] in [EVENT_LABELS['pass'], EVENT_LABELS['catch and pass'], 
                                                 EVENT_LABELS['handoff catch and pass'],
                                                 EVENT_LABELS['catch'], EVENT_LABELS['handoff catch']]:
                    true_tau = t_data_frame[SHOT_CLOCK_ID] - t_data_frame_next[SHOT_CLOCK_ID]
                    exp_tau = calc_expected_tau(t_data_frame, t_data_frame_next, e_data_frame, 
                                                e_data_frame_next, accel, r_time, vmax)
                    if true_tau > 0:
                        hist_array.append(true_tau - exp_tau)
    try:
        hist = plt.hist(hist_array, bins=int((np.max(hist_array)-np.min(hist_array))/0.4), 
                        range=None, density=True, cumulative=True)
        x = [(hist[1][i] + hist[1][i+1]) / 2 for i in np.arange(len(hist[1])-1)]
        y = hist[0]

        initial_parameter = [0.5, 0, 1]
        fit_params, _ = curve_fit(skewcauchy.cdf, x, y, p0=initial_parameter)
        plt.close()
        return fit_params, np.min(hist[1])
    
    except ValueError:
        print("fitting did not go well")
        return [0], 100

def calc_time_to_intercept(r_start, r_final, v_current, accel, r_time, vmax):
    vini = np.linalg.norm(v_current)

    adjust_position = r_start + v_current * r_time

    t = (-vini / accel + np.sqrt(vini ** 2 / accel ** 2 + 2 * np.linalg.norm(r_final - adjust_position) / accel))
    
    if vini + t * accel > vmax:
        limit_time = (vmax - vini) / accel
        remaining_distance = np.linalg.norm(r_final - adjust_position) \
                                - (vini * limit_time + 0.5 * accel * limit_time ** 2) 
        time_to_intercept = r_time + limit_time + remaining_distance / vmax
    else:
        time_to_intercept = r_time + t
        
    return time_to_intercept

def calc_expected_tau(t_data_frame, t_data_frame_next, e_data_frame, e_data_frame_next, accel, r_time, vmax):
    if e_data_frame[EVENT_LABEL] in [EVENT_LABELS['pass'], EVENT_LABELS['catch and pass'], 
                                     EVENT_LABELS['handoff catch and pass']]:
        idx = int(e_data_frame_next[BALL_PID_IDX]) - 1
    elif e_data_frame[EVENT_LABEL] in [EVENT_LABELS['catch'], EVENT_LABELS['handoff catch']]:
        idx = int(e_data_frame[BALL_PID_IDX]) - 1

    r_start = t_data_frame[PLAYER_POSITIONS][ idx*2 : idx*2+2 ]
    r_final = t_data_frame_next[PLAYER_POSITIONS][ idx*2 : idx*2+2 ]
    v_current = t_data_frame[PLAYER_VELOCITIES][ idx*2 : idx*2+2 ]

    return calc_time_to_intercept(r_start, r_final, v_current, accel, r_time, vmax)

def skewed_cauchy_distribution(x, a, loc, scale):
    bottom_bottom = scale**2 * (1 + a * np.sign(x - loc))**2
    bottom_top = (x - loc)**2
    bottom = scale * np.pi * (1 + bottom_top/bottom_bottom)
    return 1 / bottom

def get_cdf_value(x, fit_params, integral_xmin, num_points=1000):
    a, loc, scale = fit_params[0], fit_params[1], fit_params[2]
    if x <= integral_xmin:
        return 0
    else:
        x_values = np.linspace(integral_xmin, x, num_points)
        y_values = skewed_cauchy_distribution(x_values, a, loc, scale)
        return np.trapz(y_values, x_values)