from typing import Dict, Iterable, Tuple, List

import logging
import os
import traceback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..abstract_space_model import AbstractSpaceModel
from .get_obpv_value import calculate_obpv


class SoccerObpv(AbstractSpaceModel):
    """
    Class for calculating Off-Ball Positioning Value (OBPV) in soccer.

    This implementation was written originally by Yohei Ogawa (@Nagoya University); Rikuhei Umemoto (@Nagoya University).
    Please refer to the following paper for more details:
    Ogawa, Y., Umemoto, R., & Fujii, K. (2025). Space evaluation at the starting point of soccer transitions. arXiv preprint arXiv:2505.14711.s
    """

    def __init__(
            self,
            event_data=None,
            tracking_home=None,
            tracking_away=None,
            out_path=None,
            testing_mode=False
    ):
        super().__init__(
            event_data, tracking_home, tracking_away, out_path, testing_mode
        )

    def calculate(self) -> Dict[str, pd.DataFrame]:
        event_dict, tracking_home_dict, tracking_away_dict = self.read_data()
        match_ids = sorted(
            set(event_dict.keys())
            & set(tracking_home_dict.keys())
            & set(tracking_away_dict.keys())
        )

        if not match_ids:
            raise ValueError(
                "No matching keys found across event and tracking inputs.")

        if self.testing_mode:
            print("Testing mode enabled: using only the first 10 events per match.")

        results: Dict[str, pd.DataFrame] = {}
        iterator: Iterable[str] = match_ids
        iterator = tqdm(match_ids, desc="Calculating OBPV for all matches")

        failures: List[str] = []
        logger = logging.getLogger(__name__)

        for match_id in iterator:
            events_path = event_dict[match_id]
            tracking_home_path = tracking_home_dict[match_id]
            tracking_away_path = tracking_away_dict[match_id]

            try:
                results[match_id] = calculate_obpv(
                    events_path,
                    tracking_home_path,
                    tracking_away_path,
                    self.testing_mode
                )
            except Exception as e:
                tqdm.write(f"Error processing {match_id}: {e}")
                logger.exception("Error processing %s", match_id)
                tqdm.write(traceback.format_exc())
                failures.append(match_id)
                results[match_id] = pd.DataFrame()
                continue

        if failures:
            tqdm.write(f"Finished with {len(failures)} failure(s): {failures}")
            logger.warning("Finished with %d failure(s): %s",
                           len(failures), failures)
        
        if self.out_path:
            for match_id in results.keys():
                self._persist_result(match_id, results[match_id])

        return results

    def visualize(self) -> Tuple[plt.figure, plt.axes]:
        # Visualization logic for OBPV
        fig, ax = plt.subplots()
        # Add plotting code here
        return fig, ax
    
    def _persist_result(self, match_id: str, result: pd.DataFrame) -> None:
        """
        Persist the OBPV result to the output path.

        Parameters
        ----------
        match_id : str
            The match identifier.
        result : pd.DataFrame
            The OBPV result DataFrame to be saved.
        """
        assert self.out_path is not None
        data_dir = self.out_path
        data_dir.mkdir(parents=True, exist_ok=True)

        obpv_dir = data_dir / "obpv"
        obpv_dir.mkdir(parents=True, exist_ok=True)

        if not result.empty:
            result.to_pickle(obpv_dir / f"{match_id}_obpv.pkl")
            np.save(obpv_dir / f"{match_id}_obpv.npy", result.values)


        # os.makedirs(self.out_path + '/' + 'obpv', exist_ok=True)
        # result.to_pickle(self.out_path + '/' + 'obpv' + '/' + f'{match_id}_obpv.pkl')
