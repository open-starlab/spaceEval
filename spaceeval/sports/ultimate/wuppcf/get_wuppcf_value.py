from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import get_model_params, get_provider_settings
from .obso_repo import Metrica_IO as mio
from .obso_repo import Metrica_PitchControl as mpc
from .obso_repo import Metrica_Velocities as mvel


@dataclass(frozen=True)
class WUPPCFResult:
    wuppcf: np.ndarray
    player_wuppcf: np.ndarray
    events_metric: pd.DataFrame
    tracking_home_metric: pd.DataFrame
    tracking_away_metric: pd.DataFrame


def _prepare_removed_players(events: pd.DataFrame) -> pd.DataFrame:
    removed_players = pd.DataFrame(
        {
            "Frame": events["Start Frame"].to_numpy(copy=True),
            "Player": pd.to_numeric(events["From"], errors="coerce"),
        }
    )
    # Use non-inplace methods to avoid FutureWarning
    removed_players["Player"] = removed_players["Player"].ffill()
    removed_players["Player"] = removed_players["Player"].bfill()
    removed_players.loc[removed_players["Player"] == 15.0, "Player"] = np.nan
    return removed_players


def _iterate_with_progress(
    iterable: Iterable[Tuple[int, float]], use_tqdm: bool, total: int
):
    if use_tqdm:
        return tqdm(iterable, total=total, desc="Calculating wUPPCF", leave=False)
    return iterable


def calculate_wuppcf(
    events_path: str,
    tracking_home_path: str,
    tracking_away_path: str,
    provider: str = "UltimateTrack",
    use_tqdm: bool = False,
) -> WUPPCFResult:
    """
    Calculate Ultimate Frisbee weighted Pitch Control Field (wUPPCF).

    Parameters
    ----------
    events : pd.DataFrame
        Event data containing at least 'Start Frame', 'Start X', 'Start Y', and 'From'.
    tracking_home : pd.DataFrame
        Tracking data for the attacking team with frame index and player coordinates.
    tracking_away : pd.DataFrame
        Tracking data for the defending team with frame index and player coordinates.
    provider : str, optional
        Data provider name defined in the providers config file, by default "UltimateTrack".
    use_tqdm : bool, optional
        Display a tqdm progress bar when processing events, by default False.

    Returns
    -------
    WUPPCFResult
        Dataclass containing UPPCF, weighted UPPCF, per-player UPPCF, metadata, and intermediate arrays.
    """

    events = mio.read_event_data(events_path)
    tracking_home = mio.tracking_data(tracking_home_path, "Home")
    tracking_away = mio.tracking_data(tracking_away_path, "Away")

    if events.empty:
        raise ValueError("events DataFrame must not be empty.")

    settings = get_provider_settings(provider)
    params = get_model_params(provider)
    field_dimen = settings.field_dimen
    grid_size_field_dimen = (
        field_dimen[0] / params["grid_size"],
        field_dimen[1] / params["grid_size"],
    )

    # Preserve raw coordinate space for ball and stalling positions
    events_raw = events.copy()
    tracking_away_raw = tracking_away.copy()

    # Convert tracking and event data to metric coordinates for pitch control
    events_metric = mio.to_metric_coordinates(
        events.copy(), field_dimen=field_dimen, grid_size=params["grid_size"]
    )
    tracking_home_metric = mio.to_metric_coordinates(
        tracking_home.copy(), field_dimen=field_dimen, grid_size=params["grid_size"]
    )
    tracking_away_metric = mio.to_metric_coordinates(
        tracking_away.copy(), field_dimen=field_dimen, grid_size=params["grid_size"]
    )

    tracking_home_metric = mvel.calc_player_velocities(
        tracking_home_metric, smoothing=True
    )
    tracking_away_metric = mvel.calc_player_velocities(
        tracking_away_metric, smoothing=True
    )

    removed_players = _prepare_removed_players(events_metric)

    num_events = len(events_metric)
    uppcf_frames: List[Optional[np.ndarray]] = [None] * num_events
    player_frame_maps: List[Dict[str, np.ndarray]] = [{} for _ in range(num_events)]
    defending_removed = np.full(num_events, np.nan, dtype=float)

    last_attacking_players = []
    last_defending_players = []

    iterator = _iterate_with_progress(
        enumerate(events_metric["Start Frame"]), use_tqdm=use_tqdm, total=num_events
    )

    for event_idx, frame in iterator:
        if pd.isna(frame):
            continue
        (
            uppcf_frame,
            frame_player_uppcf,
            defending_removed_player,
            attacking_players,
            defending_players,
        ) = mpc.generate_pitch_control_for_event(
            event_idx,
            events_metric,
            tracking_home_metric,
            tracking_away_metric,
            removed_players,
            params,
            field_dimen=grid_size_field_dimen,
            n_grid_cells_x=int(field_dimen[0] / params["grid_size"]),
            remove=True,
        )

        uppcf_frames[event_idx] = uppcf_frame
        player_frame_maps[event_idx] = {
            str(player.id): frame_player_uppcf[player_idx]
            for player_idx, player in enumerate(attacking_players)
        }
        if defending_removed_player is not None:
            defending_removed[event_idx] = float(defending_removed_player)

        last_attacking_players = attacking_players
        last_defending_players = defending_players

    grid_shape = next(
        (frame.shape for frame in uppcf_frames if frame is not None), None
    )
    if grid_shape is None:
        raise ValueError("No valid events with non-null Start Frame were found.")

    uppcf_array = np.stack(
        [
            frame if frame is not None else np.zeros(grid_shape, dtype=float)
            for frame in uppcf_frames
        ],
        axis=0,
    )

    player_ids = sorted(
        {player_id for mapping in player_frame_maps for player_id in mapping.keys()}
    )

    if player_ids:
        player_uppcf_array = np.zeros(
            (len(player_ids), num_events, *grid_shape), dtype=float
        )
        id_to_index = {player_id: idx for idx, player_id in enumerate(player_ids)}
        for event_idx, mapping in enumerate(player_frame_maps):
            for player_id, player_grid in mapping.items():
                player_uppcf_array[id_to_index[player_id], event_idx] = player_grid
    else:
        player_uppcf_array = np.zeros((0, num_events, *grid_shape), dtype=float)

    ball_start_positions = events_raw.loc[:, ["Start X", "Start Y"]].to_numpy(
        dtype=float
    )
    stalling_positions = np.full_like(ball_start_positions, np.nan)

    tracking_away_reset = tracking_away_raw.reset_index(drop=True)
    for event_idx, player_id in enumerate(defending_removed):
        if np.isnan(player_id):
            continue
        col_x = f"Away_{int(player_id)}_x"
        col_y = f"Away_{int(player_id)}_y"
        frame_number = events_raw.iloc[event_idx]["Start Frame"]
        if (
            col_x in tracking_away_reset.columns
            and col_y in tracking_away_reset.columns
            and not pd.isna(frame_number)
        ):
            frame_number_int = int(frame_number)
            if 0 <= frame_number_int < len(tracking_away_reset):
                stalling_positions[event_idx] = tracking_away_reset.loc[
                    frame_number_int, [col_x, col_y]
                ].to_numpy(dtype=float)

    wuppcf, player_wuppcf = mpc.calculate_ultimate_pitch_control(
        uppcf_array,
        player_uppcf_array,
        ball_start_positions,
        stalling_positions,
        last_attacking_players,
        last_defending_players,
        field_dimen=grid_size_field_dimen,
    )

    return WUPPCFResult(
        wuppcf=wuppcf,
        player_wuppcf=player_wuppcf,
        events_metric=events_metric,
        tracking_home_metric=tracking_home_metric,
        tracking_away_metric=tracking_away_metric,
    )
