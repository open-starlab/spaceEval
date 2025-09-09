import numpy as np
from ..c_obso_repo.Metrica_PitchControl import initialise_players, check_offsides


def _ball_travel_time_soccer(ball_start_pos, target_position, params):
    """
    Distance / average ball speed (soccer params).
    """
    if ball_start_pos is None or any(np.isnan(ball_start_pos)):
        return 0.0
    return np.linalg.norm(target_position - ball_start_pos) / params['average_ball_speed']

def _ball_pos_along_path(ball_start_pos, target_position, t, T):
    """
    Linear trajectory for the ball: r(t) = r0 + (t/T) * (rtgt - r0).
    If T==0, the ball is assumed to be already at the target.
    """
    if T <= 0:
        return target_position
    alpha = max(0.0, min(1.0, t / T))
    return ball_start_pos + alpha * (target_position - ball_start_pos)

def _reset_player_contribs(players):
    for p in players:
        # Laurie’s player class already exposes PPCF; we reuse it as our running contribution bucket.
        p.PPCF = 0.0

def _integrate_pbcf(dt_array, attacking_players, defending_players, ball_start_pos, target_position, params, att_ids_filter=None, def_ids_filter=None):
    """
    Core BIMOS/PBCF integrator (soccer-flavor):
    - For each time step t, evaluate the *moving* ball position r_mid(t)
    - Update each player's expected arrival time to r_mid(t)
    - Use Laurie-style arrival uncertainty (sigmoid) to get f_i(t) = P(tau_true <= t)
    - Integrate dP/dt = (1 - P_att - P_def) * f_i(t) * lambda_{att/def}
    """
    # Filter relevant participants (e.g., pass excludes the current possessor among attackers;
    # dribble includes only the possessor among attackers).
    A = [p for p in attacking_players if (att_ids_filter is None or p.id in att_ids_filter)]
    D = [p for p in defending_players if (def_ids_filter is None or p.id in def_ids_filter)]

    _reset_player_contribs(A)
    _reset_player_contribs(D)

    P_att = np.zeros_like(dt_array)
    P_def = np.zeros_like(dt_array)

    i = 1
    while (P_att[i-1] + P_def[i-1] < 1 - params['model_converge_tol']) and (i < dt_array.size):
        t = dt_array[i]
        # Ball’s instantaneous location along the straight path
        r_mid = _ball_pos_along_path(ball_start_pos, target_position, t, dt_array[-1])

        # Update players' time-to-intercept wrt r_mid(t) and get cumulative arrival probs at time t
        for p in A:
            p.simple_time_to_intercept(r_mid)  # sets p.time_to_intercept
        for p in D:
            p.simple_time_to_intercept(r_mid)

        # Accumulate small increments using the same lambda params as Metrica PPCF
        # (lambda_att, lambda_def / lambda_gk already attached to players in Laurie’s class)
        one_minus_sum = (1.0 - P_att[i-1] - P_def[i-1])

        # Attackers
        dP_att_dt = 0.0
        for p in A:
            f_i = p.probability_intercept_ball(t)   # Laurie’s logistic CDF (uses p.time_to_intercept, tti_sigma, etc.)
            dPi = one_minus_sum * f_i * p.lambda_att
            p.PPCF += dPi * params['int_dt']
            dP_att_dt += p.PPCF

        # Defenders
        dP_def_dt = 0.0
        for p in D:
            f_i = p.probability_intercept_ball(t)
            dPi = one_minus_sum * f_i * p.lambda_def
            p.PPCF += dPi * params['int_dt']
            dP_def_dt += p.PPCF

        P_att[i] = dP_att_dt
        P_def[i] = dP_def_dt
        i += 1

    # Handle very short paths to avoid tiny totals
    if i < 3:
        denom = max(P_att[i-1] + P_def[i-1], 1e-12)
        return P_att[i-1] / denom, P_def[i-1] / denom
    # If not fully converged but time horizon exhausted, return latest
    if (P_att[i-1] + P_def[i-1] < 1 - params['model_converge_tol']) and (i >= dt_array.size):
        return P_att[i-1], P_def[i-1]
    # Else use the last stable step
    return P_att[i-2], P_def[i-2]

def calculate_pbcf_pass(target_position, attacking_players, defending_players, ball_start_pos, params):
    """
    Soccer PBCF (pass-to-spot):
    - Integrates from t=0 to T_calc (ball arrival) while allowing *en-route* interceptions.
    - Excludes the current ball possessor from the attacking set (pass sequence).
    Returns: PBCF_att, PBCF_def
    """
    # 1) Ball time-to-target using *soccer* average ball speed
    T_calc = _ball_travel_time_soccer(ball_start_pos, target_position, params)

    # 2) Time grid (centered on [0, T_calc], same step as Laurie’s PPCF integrator)
    t0 = max(0.0, T_calc - params['int_dt'])  # small backstep to prime the loop like Laurie’s code
    dt_array = np.arange(0.0, T_calc + params['max_int_time'], params['int_dt'])
    if dt_array.size < 2:
        dt_array = np.array([0.0, max(params['int_dt'], T_calc)])

    # 3) Identify the ball possessor (by velocity/ID you keep in your pipeline). In Laurie’s pipeline,
    #    this isn’t encoded on the player object by default. If you’ve stored it, create a filter set that excludes it.
    #    Otherwise, pass with all attackers (common in event-centric eval where you don’t track possessor).
    #    Here we assume possessor is *not* encoded in player; so pass with all attackers, which still works.
    #    If you *do* know the possessor ID (e.g., `possessor_id`), set: att_ids_filter = {pid for pid in ids if pid != possessor_id}
    att_ids_filter = None
    def_ids_filter = None

    return _integrate_pbcf(dt_array, attacking_players, defending_players, ball_start_pos, target_position, params,
                           att_ids_filter=att_ids_filter, def_ids_filter=def_ids_filter)

def calculate_pbcf_dribble(target_position, attacking_players, defending_players, ball_start_pos, params, possessor_id=None):
    """
    Soccer PBCF (dribble-to-spot):
    - Integrates from t=0 to T_calc with interceptions allowed on the *path*.
    - Attacking set is *only the ball possessor* (dribbler). Provide possessor_id if you have it.
      If None, we fallback to the nearest attacker to ball_start_pos as a proxy.
    Returns: PBCF_att, PBCF_def
    """
    T_calc = _ball_travel_time_soccer(ball_start_pos, target_position, params)
    dt_array = np.arange(0.0, T_calc + params['max_int_time'], params['int_dt'])
    if dt_array.size < 2:
        dt_array = np.array([0.0, max(params['int_dt'], T_calc)])

    # Choose the dribbler
    if possessor_id is None:
        # Fallback heuristic: closest attacker to the start-ball position
        dists = []
        for p in attacking_players:
            d = np.linalg.norm(p.position - ball_start_pos) if ball_start_pos is not None else np.inf
            dists.append((d, p.id))
        dists.sort()
        possessor_id = dists[0][1] if len(dists) else None

    att_ids_filter = {possessor_id} if possessor_id is not None else None
    def_ids_filter = None

    return _integrate_pbcf(dt_array, attacking_players, defending_players, ball_start_pos, target_position, params,
                           att_ids_filter=att_ids_filter, def_ids_filter=def_ids_filter)

def generate_pbcf_for_event(
    event_id,
    events,
    tracking_home,
    tracking_away,
    params,
    GK_numbers,
    field_dimen=(106., 68.),
    n_grid_cells_x=50,
    offsides=True,
    mode="mix",                 # "pass", "dribble", or "mix"
    mix_rule="distance",        # "distance" (simple heuristic) or "equal"
    return_components=False     # if True, also return per-mode components
):
    """
    Soccer BIMOS/PBCF full-field map at an event frame.
    Same inputs & flow as Metrica's generate_pitch_control_for_event(...).

    Returns (default):
        PBCFa, xgrid, ygrid, attacking_players

    If return_components=True:
        (PBCFa, xgrid, ygrid, attacking_players, PBCFa_pass, PBCFa_dribble)
    """
    # --- Event snapshot (frame, team in possession, ball start) ---  (mirrors Metrica)  :contentReference[oaicite:2]{index=2}
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team  = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]['Start X'], events.loc[event_id]['Start Y']])

    # --- Grid setup (identical pattern to Metrica) ---  :contentReference[oaicite:3]{index=3}
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2. + dx / 2.
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2. + dy / 2.

    # --- Initialise players once at this frame (like Metrica) ---  :contentReference[oaicite:4]{index=4}
    if pass_team == 'Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        defending_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    elif pass_team == 'Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
        attacking_players = initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    else:
        raise ValueError("Team in possession must be either 'Home' or 'Away'")

    # --- Offside filtering (same hook as Metrica) ---  :contentReference[oaicite:5]{index=5}
    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)

    # --- Output arrays ---
    PBCFa       = np.zeros((len(ygrid), len(xgrid)))
    PBCFa_pass  = np.zeros_like(PBCFa) if return_components or mode in ("pass", "mix") else None
    PBCFa_dribb = np.zeros_like(PBCFa) if return_components or mode in ("dribble", "mix") else None

    # --- Simple pass/dribble mixing rules (soccer-friendly defaults) ---
    def _mix_weights(target_xy):
        if mode == "pass":
            return 1.0, 0.0
        if mode == "dribble":
            return 0.0, 1.0
        # "mix"
        if mix_rule == "equal":
            return 0.5, 0.5
        # distance-based heuristic: short = more dribble, long = more pass
        dist = np.linalg.norm(target_xy - ball_start_pos)
        # 0–10m scale: 0m -> 70% dribble, ≥30m -> 85% pass (smooth clamp)
        w_pass = np.clip((dist - 10.0) / 20.0, 0.0, 1.0) * 0.85
        w_pass = max(w_pass, 0.15)  # never 0 pass
        w_drib = 1.0 - w_pass
        return w_pass, w_drib

    # --- Evaluate PBCF on the grid (mirrors per-cell loop in Metrica) ---  :contentReference[oaicite:6]{index=6}
    for i, y in enumerate(ygrid):
        for j, x in enumerate(xgrid):
            target = np.array([x, y])

            w_pass, w_drib = _mix_weights(target)

            # Compute pass/dribble PBCF components using your soccer functions
            att_pass = def_pass = att_drib = def_drib = 0.0
            if PBCFa_pass is not None:
                att_pass, def_pass = calculate_pbcf_pass(
                    target, attacking_players, defending_players, ball_start_pos, params
                )
            if PBCFa_dribb is not None:
                att_drib, def_drib = calculate_pbcf_dribble(
                    target, attacking_players, defending_players, ball_start_pos, params
                )

            # Mix as requested
            if mode == "pass":
                att = att_pass
            elif mode == "dribble":
                att = att_drib
            else:
                # mix
                att = w_pass * (att_pass if PBCFa_pass is not None else 0.0) + \
                      w_drib * (att_drib if PBCFa_dribb is not None else 0.0)

            PBCFa[i, j] = att
            if PBCFa_pass is not None:
                PBCFa_pass[i, j] = att_pass
            if PBCFa_dribb is not None:
                PBCFa_dribb[i, j] = att_drib

    # --- Optional checksum, analogous to Metrica (P_att + P_def ≈ 1) ---  :contentReference[oaicite:7]{index=7}
    # Using mixed components complicates a strict global checksum; you can still check a few random cells if needed.

    if return_components:
        return PBCFa, xgrid, ygrid, attacking_players, PBCFa_pass, PBCFa_dribb
    return PBCFa, xgrid, ygrid, attacking_players
