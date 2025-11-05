from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import get_model_params, get_provider_settings
from .get_wuppcf_value import WUPPCFResult, calculate_wuppcf
from .obso_repo import Metrica_Viz as mviz


class ultimate_wuppcf:
    def __init__(
        self,
        event_data=None,
        tracking_home=None,
        tracking_away=None,
        provider: str = "UltimateTrack",
        out_path: Union[str, os.PathLike, None] = None,
        testing_mode: bool = False,
        show_progress: bool = True,
    ):
        self.event_data = event_data
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.provider = provider
        self.out_path = Path(out_path) if out_path else None
        self.testing_mode = testing_mode
        self.show_progress = show_progress

    @staticmethod
    def _extract_match_key(path: Union[os.PathLike, str]) -> str:
        stem = Path(path).stem
        return "_".join(stem.split("_")[:-1])

    @staticmethod
    def _ensure_filepath_dict(source) -> Dict[str, str]:
        """
        Convert input source to a dictionary mapping match_id -> CSV file path.
        """
        if source is None:
            raise ValueError("Input source cannot be None.")

        if isinstance(source, pd.DataFrame):
            raise ValueError(
                "DataFrame input cannot be converted to file path. "
                "Please provide file path(s) or directory."
            )

        if isinstance(source, dict):
            # Assume dict contains file paths as values
            return {str(key): str(value) for key, value in source.items()}

        if isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
            files = [
                os.path.join(source, f)
                for f in os.listdir(source)
                if f.lower().endswith(".csv")
            ]
            result = {}
            for path in files:
                key = ultimate_wuppcf._extract_match_key(path)
                result[key] = str(path)
            return result

        if isinstance(source, (str, os.PathLike)) and os.path.isfile(source):
            path = Path(source)
            return {ultimate_wuppcf._extract_match_key(path): str(path)}

        raise TypeError(
            "Input must be a dict of file paths, a CSV file path, or a directory path."
        )

    def read_data(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """
        Returns dictionaries mapping match_id to CSV file paths.

        Returns:
            Tuple of three dictionaries:
            - events_dict: {match_id: event_csv_path}
            - tracking_home_dict: {match_id: home_tracking_csv_path}
            - tracking_away_dict: {match_id: away_tracking_csv_path}
        """
        events_dict = self._ensure_filepath_dict(self.event_data)
        tracking_home_dict = self._ensure_filepath_dict(self.tracking_home)
        tracking_away_dict = self._ensure_filepath_dict(self.tracking_away)
        return events_dict, tracking_home_dict, tracking_away_dict

    def get_wuppcf(self) -> Dict[str, WUPPCFResult]:
        events_dict, tracking_home_dict, tracking_away_dict = self.read_data()

        match_ids = sorted(
            set(events_dict.keys())
            & set(tracking_home_dict.keys())
            & set(tracking_away_dict.keys())
        )
        if not match_ids:
            raise ValueError("No matching keys found across event and tracking inputs.")

        if self.testing_mode:
            print("Testing mode enabled: using only the first 10 events per match.")

        results: Dict[str, WUPPCFResult] = {}
        iterator: Iterable[str] = match_ids
        if self.show_progress and len(match_ids) > 1:
            iterator = tqdm(match_ids, desc="Matches", leave=False)

        for match_id in iterator:
            # Load DataFrames from file paths
            events_path = events_dict[match_id]
            tracking_home_path = tracking_home_dict[match_id]
            tracking_away_path = tracking_away_dict[match_id]

            result = calculate_wuppcf(
                events_path,
                tracking_home_path,
                tracking_away_path,
                provider=self.provider,
                use_tqdm=self.show_progress and len(match_ids) == 1,
            )
            results[match_id] = result

            if self.out_path:
                self._persist_result(match_id, result)

        return results

    def _persist_result(self, match_id: str, result: WUPPCFResult) -> None:
        assert self.out_path is not None
        data_dir = self.out_path / "wuppcf"
        data_dir.mkdir(parents=True, exist_ok=True)

        np.save(data_dir / f"{match_id}_wUPPCF.npy", result.wuppcf)
        np.save(data_dir / f"{match_id}_player_wUPPCF.npy", result.player_wuppcf)

    def vis_wuppcf(
        self,
        results: Dict[str, WUPPCFResult],
    ):
        settings = get_provider_settings(self.provider)
        params = get_model_params(self.provider)
        field_dimen = settings.field_dimen

        for match_id, result in results.items():
            mviz.save_match_clip_OBSO(
                hometeam=result.tracking_home_metric,
                awayteam=result.tracking_away_metric,
                wUPPCF=result.wuppcf,
                fpath=self.out_path if self.out_path else ".",
                fname=f"{match_id}_wUPPCF",
                figax=None,
                frames_per_second=settings.fps,
                field_dimen=field_dimen,
                include_player_velocities=True,
                PlayerMarkerSize=10,
                vmin=0,
                vmax=1,
                colorbar=True,
                cm="Blues",
                grid_size=params["grid_size"],
            )
