from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

DataDictionaryType = Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]


class AbstractSpaceModel(ABC):

    MATCH_ID_REGEX = r'^(?:event_data|home_tracking|away_tracking)_(.+)$'

    def __init__(
        self,
        event_data_path=None,
        tracking_home_path=None,
        tracking_away_path=None,
        out_path: Union[str, os.PathLike, None] = None,
        testing_mode=False
    ):
        self.testing_mode = testing_mode
        self.event_data, self.tracking_home_data, self.tracking_away_data = self.read_data(
            event_data_path, tracking_home_path, tracking_away_path)
        self.out_path = Path(out_path) if out_path else None

    @abstractmethod
    def calculate(self) -> Dict[str, pd.DataFrame]:
        """
        Abstract method to calculate space model metrics.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping match_id to DataFrame of calculated metrics.
        """
        raise NotImplementedError("`calculate` method is not implemented.")

    @abstractmethod
    def visualize(self) -> Tuple[plt.figure, plt.axes]:
        """
        Abstract method to visualize space model metrics.

        Returns
        -------
        Tuple[plt.figure, plt.axes]
            Matplotlib figure and axes containing the visualization.
        """
        raise NotImplementedError("`visualize` method is not implemented.")

    def read_data(self, event_data_path, tracking_home_path, tracking_away_path) -> DataDictionaryType:
        """
        Return dictionaries mapping match_id to CSV file paths.

        Parameters
        ----------

        Returns
        -------
        DataDictionaryType
            Tuple of three dictionaries:
            - events_dict: {match_id: event_csv_path}
            - tracking_home_dict: {match_id: home_tracking_csv_path}
            - tracking_away_dict: {match_id: away_tracking_csv_path}
        """
        events_dict = self._ensure_filepath_dict(event_data_path)
        tracking_home_dict = self._ensure_filepath_dict(tracking_home_path)
        tracking_away_dict = self._ensure_filepath_dict(tracking_away_path)

        match_ids = sorted(
            set(events_dict.keys())
            & set(tracking_home_dict.keys())
            & set(tracking_away_dict.keys())
        )

        if not match_ids:
            raise ValueError(
                "No matching keys found across event and tracking inputs.")

        if self.testing_mode:
            print("In testing mode, only up to 5 files per match will be read.")
            match_ids = match_ids[:5]

        events_dfs = {k: pd.read_csv(
            events_dict[k]) for k in tqdm(match_ids, desc="Reading event data")}
        tracking_home_dfs = {k: pd.read_csv(
            tracking_home_dict[k]) for k in tqdm(match_ids, desc="Reading home tracking data")}
        tracking_away_dfs = {k: pd.read_csv(
            tracking_away_dict[k]) for k in tqdm(match_ids, desc="Reading away tracking data")}
        return events_dfs, tracking_home_dfs, tracking_away_dfs

    @staticmethod
    def _ensure_filepath_dict(source) -> Dict[str, str]:
        """
        Ensure the input source is a dictionary mapping match IDs to file paths.

        Parameters
        ----------
        source : Union[str, os.PathLike, Dict[str, str]]
            Input source which can be a directory path, file path, or dictionary.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping match_id to file paths.

        Raises
        ------
        ValueError
            If the input source is None or a DataFrame.
        TypeError
            If the input source type is unsupported.
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
                key = AbstractSpaceModel._extract_match_key(path)
                result[key] = str(path)
            return result

        if isinstance(source, (str, os.PathLike)) and os.path.isfile(source):
            path = Path(source)
            return {AbstractSpaceModel._extract_match_key(path): str(path)}

        raise TypeError(
            "Input must be a dict of file paths, a CSV file path, or a directory path."
        )

    @staticmethod
    def _extract_match_key(path: Union[os.PathLike, str]) -> str:
        stem = Path(path).stem
        match = re.match(AbstractSpaceModel.MATCH_ID_REGEX, stem)
        if match:
            return match.group(1)
        return stem
