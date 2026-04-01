from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


class VFrameResult:
    """Container for v_frame calculation results."""

    def __init__(self, scenario_id: str, v_frame_data: pd.DataFrame):
        self.scenario_id = scenario_id
        self.v_frame_data = v_frame_data


class VScenarioResult:
    """Container for v_scenario calculation results."""

    def __init__(self, play_id: str, v_scenario_data: pd.DataFrame):
        self.play_id = play_id
        self.v_scenario_data = v_scenario_data


class ultimate_crsv:
    def __init__(
        self,
        wuppcf_path: Union[str, os.PathLike],
        scenario_path: Union[str, os.PathLike],
        out_path: Union[str, os.PathLike, None] = None,
        provider: str = "UltimateTrack",
        disc_speed: float = 15.44,
        testing_mode: bool = False,
    ):
        """
        Initialize the CRSV (Counterfactual Reasoning for Space Value) calculator.

        Args:
            wuppcf_path: Path to wUPPCF results directory
            scenario_path: Path to scenario (counterfactual) data files
            out_path: Output path for saving results
            provider: Data provider name ("UltimateTrack" or "UFATrack")
            disc_speed: Speed of the disc (default: 15.44)
            testing_mode: If True, process only first few files for testing
        """
        self.wuppcf_path = Path(wuppcf_path)
        self.scenario_path = Path(scenario_path)
        self.out_path = Path(out_path) if out_path else None
        self.provider = provider
        self.disc_speed = disc_speed
        self.testing_mode = testing_mode

    def _get_grid_config(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get grid configuration based on provider."""
        if self.provider == "UltimateTrack":
            xgrid = np.arange(1, 94, 2)
            ygrid = np.arange(37 / 18 / 2, 37, 37 / 18)
        elif self.provider == "UFATrack":
            xgrid = np.arange(110 / 54 / 2, 110, 110 / 54)
            ygrid = np.arange(49 / 24 / 2, 49, 49 / 24)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        return xgrid, ygrid

    @staticmethod
    def _calc_v_frame_single(
        wuppcf: np.ndarray,
        row: pd.Series,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
        disc_pos: Tuple[float, float],
        disc_speed: float = 15.44,
    ) -> Tuple[float, int]:
        """
        Calculate v_frame value for a single player.

        Args:
            wuppcf: 2D wUPPCF array
            row: Player data row
            xgrid: X grid coordinates
            ygrid: Y grid coordinates
            disc_pos: Disc position (x, y)
            disc_speed: Disc speed

        Returns:
            Tuple of (v_frame value, mask size)
        """
        x_p = float(row["x"])
        y_p = float(row["y"])
        vx = float(row["vx"])
        vy = float(row["vy"])
        x_d, y_d = disc_pos

        A = vx**2 + vy**2 - disc_speed**2
        B = 2 * (vx * (x_p - x_d) + vy * (y_p - y_d))
        C = (x_p - x_d) ** 2 + (y_p - y_d) ** 2

        discriminant = B**2 - 4 * A * C

        if not np.isfinite(discriminant) or discriminant < 0 or np.isclose(A, 0.0):
            return 0.0, 0

        t1 = (-B + np.sqrt(discriminant)) / (2 * A)
        t2 = (-B - np.sqrt(discriminant)) / (2 * A)
        t_candidates = [t for t in (t1, t2) if np.isfinite(t) and t > 0]
        if not t_candidates:
            return 0.0, 0
        t = max(t_candidates)

        x_t = x_p + vx * t
        y_t = y_p + vy * t

        radius = float(np.sqrt((vx * t * 0.5) ** 2 + (vy * t * 0.5) ** 2))
        if not np.isfinite(radius) or radius <= 0:
            return 0.0, 0

        # Create a 2D grid of coordinates
        X, Y = np.meshgrid(xgrid, ygrid)

        # Create a mask for points within the radius
        mask = (x_t - X) ** 2 + (y_t - Y) ** 2 < (radius) ** 2
        mask_size = int(np.sum(mask))

        # Check if the mask results in any valid points
        if mask_size == 0:
            return 0.0, 0

        # Calculate the value based on the mean value within the mask
        value = float(np.mean(np.flipud(wuppcf)[mask]).round(3))

        return value, mask_size

    def calc_v_frame(self) -> Dict[str, VFrameResult]:
        """
        Calculate v_frame values for all scenarios.

        Returns:
            Dictionary mapping scenario_id to VFrameResult
        """
        xgrid, ygrid = self._get_grid_config()
        results: Dict[str, VFrameResult] = {}

        # Find all scenario files
        scenario_files = [
            f for f in os.listdir(self.scenario_path) if f.lower().endswith(".csv")
        ]

        if self.testing_mode:
            scenario_files = scenario_files[:3]

        iterator = tqdm(scenario_files, desc="Calculating v_frame", leave=False)

        for scenario_file in iterator:
            scenario_id = scenario_file.replace(".csv", "")
            scenario_path = self.scenario_path / scenario_file

            # Find corresponding wUPPCF file
            wuppcf_file = self.wuppcf_path / f"{scenario_id}.npy"
            if not wuppcf_file.exists():
                continue

            # try:
            scenario = pd.read_csv(scenario_path)
            wuppcf = np.load(wuppcf_file)

            unique_frames = sorted(scenario["frame"].unique())
            v_frame_results = []

            for frame_idx, frame in enumerate(unique_frames):
                frame_data = scenario[scenario["frame"] == frame]

                # Get offensive players without disc
                offense = (
                    frame_data[
                        (frame_data["class"] == "offense")
                        & (frame_data["holder"] == False)
                    ]
                    .sort_values(by="id")
                    .reset_index(drop=True)
                )

                disc_pos = frame_data[frame_data["id"] == 15][["x", "y"]].values[0]
                offense_id = offense["id"].values
                v_frame_list = []

                for index, row in offense.iterrows():
                    # try:
                    v_frame, mask_size = self._calc_v_frame_single(
                        wuppcf[index, frame_idx, :, :],
                        row,
                        xgrid,
                        ygrid,
                        disc_pos,
                        self.disc_speed,
                    )
                    # except (IndexError, ValueError):
                    #     v_frame, mask_size = 0.0, 0

                    v_frame_list.append(v_frame)

                # Get selected player
                selected_id = frame_data[
                    (frame_data["selected"] == True) & (frame_data["holder"] == False)
                ]["id"].values

                if len(selected_id) == 0:
                    v_frame = 0.0
                    rank = np.nan
                else:
                    selected_num = np.where(offense_id == selected_id[0])[0]
                    if len(selected_num) > 0:
                        v_frame = v_frame_list[selected_num[0]]
                        v_frame_list_sorted = sorted(v_frame_list, reverse=True)
                        indices = [
                            i
                            for i, val in enumerate(v_frame_list_sorted)
                            if val == v_frame
                        ]
                        rank = max(indices) + 1 if indices else np.nan
                    else:
                        v_frame = 0.0
                        rank = np.nan

                v_frame_results.append(
                    {
                        "frame": frame,
                        "value": v_frame,
                        "rank": rank,
                        "other_values": v_frame_list,
                    }
                )

            v_frame_df = pd.DataFrame(v_frame_results)
            v_frame_df = v_frame_df.sort_values(by="frame")

            result = VFrameResult(scenario_id, v_frame_df)
            results[scenario_id] = result

            if self.out_path:
                self._persist_v_frame(scenario_id, v_frame_df)

            # except Exception as e:
            #     print(f"Error processing {scenario_id}: {e}")
            #     continue

        return results

    def _persist_v_frame(self, scenario_id: str, v_frame_data: pd.DataFrame) -> None:
        """
        Persist v_frame results to disk.

        Args:
            scenario_id: Scenario identifier
            v_frame_data: DataFrame with v_frame results
        """
        assert self.out_path is not None
        v_frame_dir = self.out_path / "v_frame"
        v_frame_dir.mkdir(parents=True, exist_ok=True)

        output_file = v_frame_dir / f"{scenario_id}.csv"
        v_frame_data.to_csv(output_file, index=False)

    def calc_v_scenario(
        self, v_frame_results: Dict[str, VFrameResult] | None = None
    ) -> Dict[str, VScenarioResult]:
        """
        Calculate v_scenario values from v_frame data.

        Args:
            v_frame_results: Dictionary of VFrameResult objects. If None, reads from disk.

        Returns:
            Dictionary mapping play_id to VScenarioResult
        """
        results: Dict[str, VScenarioResult] = {}

        if v_frame_results is None:
            # Read from disk if not provided
            v_frame_dir = self.out_path / "v_frame" if self.out_path else None
            if v_frame_dir is None or not v_frame_dir.exists():
                raise ValueError(
                    "v_frame_dir not found. Please run calc_v_frame() first."
                )

            v_frame_files = [
                f for f in os.listdir(v_frame_dir) if f.lower().endswith(".csv")
            ]
        else:
            v_frame_files = list(v_frame_results.keys())

        # Group by play_id
        plays_dict = {}
        for scenario_id in v_frame_files:
            if isinstance(scenario_id, str):
                play_id = "_".join(scenario_id.split("_")[:-1])
            else:
                play_id = scenario_id

            if play_id not in plays_dict:
                plays_dict[play_id] = []
            plays_dict[play_id].append(scenario_id)

        for play_id, scenario_ids in plays_dict.items():
            v_scenario_results = []

            for scenario_id in scenario_ids:
                # Extract shift value
                parts = scenario_id.split("_")
                shift = int(parts[-1])

                if v_frame_results is not None:
                    if scenario_id not in v_frame_results:
                        continue
                    v_frame_data = v_frame_results[scenario_id].v_frame_data
                else:
                    v_frame_file = v_frame_dir / f"{scenario_id}.csv"
                    if not v_frame_file.exists():
                        continue
                    v_frame_data = pd.read_csv(v_frame_file)

                # Calculate moving average
                v_frame_data["value_smoothed"] = (
                    np.convolve(v_frame_data["value"], np.ones(10), mode="same") / 10
                )
                max_idx = v_frame_data["value_smoothed"].idxmax()
                max_val = v_frame_data["value_smoothed"].max()

                v_scenario_results.append(
                    {
                        "play": play_id,
                        "shift": shift,
                        "v_scenario": max_val.round(3),
                        "max_index": max_idx,
                    }
                )

            if v_scenario_results:
                v_scenario_df = pd.DataFrame(v_scenario_results)
                v_scenario_df = v_scenario_df.sort_values(by="shift")

                result = VScenarioResult(play_id, v_scenario_df)
                results[play_id] = result

                if self.out_path:
                    self._persist_v_scenario(play_id, v_scenario_df)

        return results

    def _persist_v_scenario(self, play_id: str, v_scenario_data: pd.DataFrame) -> None:
        """
        Persist v_scenario results to disk.

        Args:
            play_id: Play identifier
            v_scenario_data: DataFrame with v_scenario results
        """
        assert self.out_path is not None
        v_scenario_dir = self.out_path / "v_scenario"
        v_scenario_dir.mkdir(parents=True, exist_ok=True)

        output_file = v_scenario_dir / f"{play_id}.csv"
        v_scenario_data.to_csv(output_file, index=False)

    def calc_v_timing(
        self, v_scenario_results: Dict[str, VScenarioResult] | None = None
    ) -> pd.DataFrame:
        """
        Calculate v_timing (Counterfactual Space Value) values.

        Args:
            v_scenario_results: Dictionary of VScenarioResult objects. If None, reads from disk.

        Returns:
            DataFrame with v_timing results
        """
        results = []

        if v_scenario_results is None:
            # Read from disk if not provided
            v_scenario_dir = self.out_path / "v_scenario" if self.out_path else None
            if v_scenario_dir is None or not v_scenario_dir.exists():
                raise ValueError(
                    "v_scenario_dir not found. Please run calc_v_scenario() first."
                )

            v_scenario_files = [
                f for f in os.listdir(v_scenario_dir) if f.lower().endswith(".csv")
            ]

            for v_scenario_file in v_scenario_files:
                play_id = v_scenario_file.replace(".csv", "")
                v_scenario_path = v_scenario_dir / v_scenario_file
                data = pd.read_csv(v_scenario_path)

                actual_v_scenario = data.loc[data["shift"] == 0, "v_scenario"].values
                if len(actual_v_scenario) == 0:
                    continue

                actual_v_scenario = actual_v_scenario[0]
                best_v_scenario = data.loc[data["shift"] != 0, "v_scenario"].max()

                best_row = data.loc[data["v_scenario"] == best_v_scenario]

                results.append(
                    {
                        "play": play_id,
                        "v_timing": (actual_v_scenario - best_v_scenario).round(3),
                        "best_shift": (
                            best_row["shift"].values[0] if len(best_row) > 0 else np.nan
                        ),
                        "actual_max_idx": (
                            data.loc[data["shift"] == 0, "max_index"].values[0]
                            if len(data.loc[data["shift"] == 0]) > 0
                            else np.nan
                        ),
                        "best_max_idx": (
                            best_row["max_index"].values[0]
                            if len(best_row) > 0
                            else np.nan
                        ),
                    }
                )
        else:
            for play_id, v_scenario_result in v_scenario_results.items():
                data = v_scenario_result.v_scenario_data

                actual_v_scenario = data.loc[data["shift"] == 0, "v_scenario"].values
                if len(actual_v_scenario) == 0:
                    continue

                actual_v_scenario = actual_v_scenario[0]
                best_v_scenario = data.loc[data["shift"] != 0, "v_scenario"].max()

                best_row = data.loc[data["v_scenario"] == best_v_scenario]

                results.append(
                    {
                        "play": play_id,
                        "v_timing": (actual_v_scenario - best_v_scenario).round(3),
                        "best_shift": (
                            best_row["shift"].values[0] if len(best_row) > 0 else np.nan
                        ),
                        "actual_max_idx": (
                            data.loc[data["shift"] == 0, "max_index"].values[0]
                            if len(data.loc[data["shift"] == 0]) > 0
                            else np.nan
                        ),
                        "best_max_idx": (
                            best_row["max_index"].values[0]
                            if len(best_row) > 0
                            else np.nan
                        ),
                    }
                )

        v_timing_df = pd.DataFrame(
            results,
            columns=[
                "play",
                "v_timing",
                "best_shift",
                "actual_max_idx",
                "best_max_idx",
            ],
        )
        v_timing_df = v_timing_df.sort_values(by="play")

        if self.out_path:
            self._persist_v_timing(v_timing_df)

        return v_timing_df

    def _persist_v_timing(self, v_timing_data: pd.DataFrame) -> None:
        """
        Persist v_timing results to disk.

        Args:
            v_timing_data: DataFrame with v_timing results
        """
        assert self.out_path is not None
        self.out_path.mkdir(parents=True, exist_ok=True)

        output_file = self.out_path / "v_timing.csv"
        v_timing_data.to_csv(output_file, index=False)

    def calc_all(
        self,
    ) -> Tuple[Dict[str, VFrameResult], Dict[str, VScenarioResult], pd.DataFrame]:
        """
        Execute the complete v_timing calculation pipeline.

        Returns:
            Tuple of (v_frame_results, v_scenario_results, v_timing_results)
        """
        print("Calculating v_frame values...")
        v_frame_results = self.calc_v_frame()
        print(
            f"v_frame calculation completed: {len(v_frame_results)} scenarios processed"
        )

        print("Calculating v_scenario values...")
        v_scenario_results = self.calc_v_scenario(v_frame_results)
        print(
            f"v_scenario calculation completed: {len(v_scenario_results)} plays processed"
        )

        print("Calculating v_timing values...")
        v_timing_results = self.calc_v_timing(v_scenario_results)
        print(
            f"v_timing calculation completed: {len(v_timing_results)} plays processed"
        )

        return v_frame_results, v_scenario_results, v_timing_results
