from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt

DataDictionaryType = Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]


class AbstractSpaceModel(ABC):

    MATCH_ID_REGEX = r'^(?:event_data|home_tracking|away_tracking)_(.+)$'

    def __init__(
        self,
        event_data=None,
        tracking_home=None,
        tracking_away=None,
        out_path: Union[str, os.PathLike, None] = None,
        testing_mode=False
    ):
        self.event_data = event_data
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.out_path = Path(out_path) if out_path else None
        self.testing_mode = testing_mode

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

    def read_data(self) -> DataDictionaryType:
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
        events_dict = self._ensure_filepath_dict(self.event_data)
        tracking_home_dict = self._ensure_filepath_dict(self.tracking_home)
        tracking_away_dict = self._ensure_filepath_dict(self.tracking_away)
        return events_dict, tracking_home_dict, tracking_away_dict

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
