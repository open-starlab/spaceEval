from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import get_provider_settings
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
    ):
        self.event_data = event_data
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.provider = provider
        self.out_path = Path(out_path) if out_path else None
        self.testing_mode = testing_mode

    @staticmethod
    def _extract_match_key(path: Union[os.PathLike, str]) -> str:
        stem = Path(path).stem
        return "_".join(stem.split("_")[:-1])

    @staticmethod
    def _ensure_filepath_dict(source) -> Dict[str, str]:
        """
        Ensure the input source is a dictionary mapping match IDs to file paths.
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
        """
        Calculate wUPPCF for all matches based on provided event and tracking data.

        Returns:
            Dictionary mapping match_id to WUPPCFResult.
        """
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
            )
            results[match_id] = result

            if self.out_path:
                self._persist_result(match_id, result)

        return results

    def _persist_result(self, match_id: str, result: WUPPCFResult) -> None:
        """
        Persist wUPPCF results to disk.

        Args:
            match_id: Match identifier
            result: WUPPCFResult object
        """
        assert self.out_path is not None
        data_dir = self.out_path
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories if they don't exist
        wuppcf_dir = data_dir / "wUPPCF"
        player_wuppcf_dir = data_dir / "player_wUPPCF"

        wuppcf_dir.mkdir(parents=True, exist_ok=True)
        player_wuppcf_dir.mkdir(parents=True, exist_ok=True)

        np.save(wuppcf_dir / f"{match_id}.npy", result.wuppcf)
        np.save(player_wuppcf_dir / f"{match_id}.npy", result.player_wuppcf)

    def vis_wuppcf(
        self,
        results: Dict[str, WUPPCFResult],
    ) -> None:
        """
        Visualize wUPPCF results.

        Args:
            results: Dictionary mapping match_id to WUPPCFResult
        """
        settings = get_provider_settings(self.provider)

        for match_id, result in results.items():
            # Create video directory if it doesn't exist
            video_dir = self.out_path / "video" if self.out_path else Path(".")
            video_dir.mkdir(parents=True, exist_ok=True)

            mviz.save_match_clip_OBSO(
                hometeam=result.tracking_home_metric,
                awayteam=result.tracking_away_metric,
                wUPPCF=result.wuppcf,
                fpath=self.out_path / "video" if self.out_path else ".",
                fname=match_id,
                figax=None,
                frames_per_second=settings.fps,
                field_dimen=settings.field_dimen,
                PlayerMarkerSize=10,
                vmin=0,
                vmax=1,
                colorbar=True,
                cm="Blues",
            )
