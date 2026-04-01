from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


class CounterfactualResult:
    """Container for counterfactual scenario generation results."""

    def __init__(self, match_id: str, scenarios: Dict[int, pd.DataFrame]):
        self.match_id = match_id
        self.scenarios = scenarios


class ultimate_generate_counterfactual:
    def __init__(
        self,
        input_data=None,
        provider: str = "UltimateTrack",
        slide_size_range: Tuple[int, int, int] | None = None,
        out_path: Union[str, os.PathLike, None] = None,
        testing_mode: bool = False,
    ):
        """
        Initialize the counterfactual scenario generator.

        Args:
            input_data: Input data source (file path, directory, or DataFrame)
            provider: Data provider name ("UltimateTrack" or "UFATrack")
            slide_size_range: Range of slide sizes (start, stop, step) for scenario generation.
                             If None, automatically set based on provider:
                             - UltimateTrack: (-15, 16, 1)
                             - UFATrack: (-10, 11, 1)
            out_path: Output path for saving results
            testing_mode: If True, process only first few files for testing
        """
        self.input_data = input_data
        self.provider = provider
        self.out_path = Path(out_path) if out_path else None
        self.testing_mode = testing_mode

        # Set slide_size_range based on provider if not provided
        if slide_size_range is None:
            if provider == "UltimateTrack":
                self.slide_size_range = (-15, 16, 1)
            elif provider == "UFATrack":
                self.slide_size_range = (-10, 11, 1)
            else:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    "Supported providers: 'UltimateTrack', 'UFATrack'"
                )
        else:
            self.slide_size_range = slide_size_range

    @staticmethod
    def _extract_match_key(path: Union[os.PathLike, str]) -> str:
        """Extract match ID from file path."""
        stem = Path(path).stem
        return stem

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
                key = ultimate_generate_counterfactual._extract_match_key(path)
                result[key] = str(path)
            return result

        if isinstance(source, (str, os.PathLike)) and os.path.isfile(source):
            path = Path(source)
            return {
                ultimate_generate_counterfactual._extract_match_key(path): str(path)
            }

        raise TypeError(
            "Input must be a dict of file paths, a CSV file path, or a directory path."
        )

    def read_data(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping match_id to CSV file paths.

        Returns:
            Dictionary: {match_id: input_csv_path}
        """
        return self._ensure_filepath_dict(self.input_data)

    @staticmethod
    def _slide_movement_forward(
        df: pd.DataFrame, slide_size: int
    ) -> pd.DataFrame | None:
        """
        Slide the movement data of the selected player forward.

        Args:
            df: DataFrame containing the data
            slide_size: The slide size (negative value for forward movement)

        Returns:
            DataFrame with shifted data, or None if operation not possible
        """
        # Get the start and end frame of the selected player
        selected_rows = df[df["selected"]]
        if len(selected_rows) == 0:
            print("警告: 選択されたプレイヤーが見つかりません。")
            return None

        start_frame = selected_rows["frame"].min()
        max_frame = df["frame"].max()
        new_start_frame = start_frame + slide_size
        new_max_frame = max_frame + slide_size

        # Check if it is possible to slide the movement data forward
        if start_frame < abs(slide_size):
            return None

        class_types = ["offense", "defense"]

        for class_type in class_types:
            # Selected player
            column = "selected" if class_type == "offense" else "def_selected"

            # Get the selected player's id
            selected_players = df[df[column]]["id"].values
            if len(selected_players) == 0:
                print(
                    f"警告: {class_type}で選択されたプレイヤーが見つかりません。スキップします。"
                )
                continue
            selected_id = int(selected_players[0])

            # Get the x and y shift
            x_shift = (
                df[(df["id"] == selected_id) & (df["frame"] == new_start_frame)][
                    "x"
                ].values[0]
                - df[(df["id"] == selected_id) & (df["frame"] == start_frame)][
                    "x"
                ].values[0]
            )
            y_shift = (
                df[(df["id"] == selected_id) & (df["frame"] == new_start_frame)][
                    "y"
                ].values[0]
                - df[(df["id"] == selected_id) & (df["frame"] == start_frame)][
                    "y"
                ].values[0]
            )

            # Remove duplicate frames by shifting the frame
            df = df[
                ~(
                    (df["id"] == selected_id)
                    & (df["frame"] >= new_start_frame)
                    & (df["frame"] < start_frame)
                )
            ]

            # Shift the frame
            df.loc[
                (df["id"] == selected_id) & (df["frame"] >= start_frame), "frame"
            ] += slide_size

            # Shift the x and y center
            df.loc[
                (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "x"
            ] = (
                df.loc[
                    (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "x"
                ]
                + x_shift
            ).round(
                2
            )
            df.loc[
                (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "y"
            ] = (
                df.loc[
                    (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "y"
                ]
                + y_shift
            ).round(
                2
            )

            # Get other features
            vx_mean = (
                df[(df["id"] == selected_id) & (df["frame"] > new_max_frame - 15)]["vx"]
                .mean()
                .round(2)
            )
            vy_mean = (
                df[(df["id"] == selected_id) & (df["frame"] > new_max_frame - 15)]["vy"]
                .mean()
                .round(2)
            )
            v_mag = np.sqrt(vx_mean**2 + vy_mean**2).round(2)
            v_angle = np.degrees(np.arctan2(vy_mean, vx_mean)).round(2)

            closest = (
                df[df["id"] == selected_id]["closest"].values[0]
                if class_type == "offense"
                else 0
            )

            # Add missing frames by shifting the frame
            columns = df.columns
            rows_to_add = []
            for i in range(1, abs(slide_size) + 1):
                x = (
                    df[(df["id"] == selected_id) & (df["frame"] == new_max_frame)][
                        "x"
                    ].values[0]
                    + vx_mean / 15 * i
                ).round(2)
                y = (
                    df[(df["id"] == selected_id) & (df["frame"] == new_max_frame)][
                        "y"
                    ].values[0]
                    + vy_mean / 15 * i
                ).round(2)

                rows_to_add.append(
                    [
                        new_max_frame + i,
                        selected_id,
                        x,
                        y,
                        vx_mean,
                        vy_mean,
                        None,
                        None,
                        v_mag,
                        None,
                        v_angle,
                        None,
                        None,
                        None,
                        None,
                        class_type,
                        False,
                        closest,
                        False,
                        False,
                        False,
                    ]
                )

            if rows_to_add:
                df_add = pd.DataFrame(rows_to_add, columns=columns)
                # Ensure consistent dtypes with original dataframe
                for col in df.columns:
                    if col in df_add.columns and df[col].dtype != object:
                        df_add[col] = df_add[col].astype(df[col].dtype)
                df = pd.concat([df, df_add], ignore_index=True)

        # Sort the dataframe
        df = df.sort_values(["frame", "id"]).reset_index(drop=True)

        return df

    @staticmethod
    def _slide_movement_backward(
        df: pd.DataFrame, slide_size: int
    ) -> pd.DataFrame | None:
        """
        Slide the movement data of the selected player backward.

        Args:
            df: DataFrame containing the data
            slide_size: The slide size (positive value for backward movement)

        Returns:
            DataFrame with shifted data, or None if operation not possible
        """
        # Get the start and end frame of the selected player
        selected_rows = df[df["selected"]]
        if len(selected_rows) == 0:
            print("警告: 選択されたプレイヤーが見つかりません。")
            return None

        start_frame = selected_rows["frame"].min()
        end_frame = selected_rows["frame"].max()
        max_frame = df["frame"].max()
        new_start_frame = start_frame + slide_size

        # Check if it is possible to slide the movement data backward
        if max_frame - end_frame < slide_size:
            return None

        class_types = ["offense", "defense"]

        for class_type in class_types:
            # Selected player
            column = "selected" if class_type == "offense" else "def_selected"

            # Get the selected player's id
            selected_players = df[df[column]]["id"].values
            if len(selected_players) == 0:
                print(
                    f"警告: {class_type}で選択されたプレイヤーが見つかりません。スキップします。"
                )
                continue
            selected_id = int(selected_players[0])

            # Remove unnecessary frames by shifting the frame
            df = df[
                ~(
                    (df["id"] == selected_id)
                    & (df["frame"] > max_frame - abs(slide_size))
                )
            ]

            # Shift the frame
            df.loc[
                (df["id"] == selected_id) & (df["frame"] >= start_frame), "frame"
            ] += slide_size

            # Get other features
            vx_mean = (
                df[
                    (df["id"] == selected_id)
                    & (df["frame"].isin(range(start_frame - 15, start_frame + 1)))
                ]["vx"]
                .mean()
                .round(2)
            )
            vy_mean = (
                df[
                    (df["id"] == selected_id)
                    & (df["frame"].isin(range(start_frame - 15, start_frame + 1)))
                ]["vy"]
                .mean()
                .round(2)
            )

            v_mag = np.sqrt(vx_mean**2 + vy_mean**2).round(2)
            v_angle = np.degrees(np.arctan2(vy_mean, vx_mean)).round(2)
            closest = (
                df[df["id"] == selected_id]["closest"].values[0]
                if class_type == "offense"
                else 0
            )

            # Add missing frames by shifting the frame
            columns = df.columns
            rows_to_add = []
            for i in range(1, slide_size + 1):
                x = (
                    df[(df["id"] == selected_id) & (df["frame"] == start_frame - 1)][
                        "x"
                    ].values[0]
                    + vx_mean / 15 * i
                ).round(2)
                y = (
                    df[(df["id"] == selected_id) & (df["frame"] == start_frame - 1)][
                        "y"
                    ].values[0]
                    + vy_mean / 15 * i
                ).round(2)

                rows_to_add.append(
                    [
                        start_frame - 1 + i,
                        selected_id,
                        x,
                        y,
                        vx_mean,
                        vy_mean,
                        None,
                        None,
                        v_mag,
                        None,
                        v_angle,
                        None,
                        None,
                        None,
                        None,
                        class_type,
                        False,
                        closest,
                        False,
                        False,
                        False,
                    ]
                )

            if rows_to_add:
                df_add = pd.DataFrame(rows_to_add, columns=columns)
                # Ensure consistent dtypes with original dataframe
                for col in df.columns:
                    if col in df_add.columns and df[col].dtype != object:
                        df_add[col] = df_add[col].astype(df[col].dtype)
                df = pd.concat([df, df_add], ignore_index=True)

            # Get the x and y shift
            x_shift = (
                df[(df["id"] == selected_id) & (df["frame"] == new_start_frame - 1)][
                    "x"
                ].values[0]
                - df[(df["id"] == selected_id) & (df["frame"] == start_frame - 1)][
                    "x"
                ].values[0]
            )
            y_shift = (
                df[(df["id"] == selected_id) & (df["frame"] == new_start_frame - 1)][
                    "y"
                ].values[0]
                - df[(df["id"] == selected_id) & (df["frame"] == start_frame - 1)][
                    "y"
                ].values[0]
            )

            # Shift the x and y center
            df.loc[
                (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "x"
            ] = (
                df.loc[
                    (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "x"
                ]
                + x_shift
            ).round(
                2
            )
            df.loc[
                (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "y"
            ] = (
                df.loc[
                    (df["id"] == selected_id) & (df["frame"] >= new_start_frame), "y"
                ]
                + y_shift
            ).round(
                2
            )

        # Sort the dataframe
        df = df.sort_values(["frame", "id"]).reset_index(drop=True)

        return df

    @staticmethod
    def _recalculate_disc_position(df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate the disc position based on holder information.

        Args:
            df: DataFrame containing the data

        Returns:
            DataFrame with recalculated disc positions
        """
        df = df.sort_values(by="frame", ascending=False).reset_index(drop=True)

        temp = None
        for frame, frame_data in df.groupby("frame"):
            holder_data = frame_data[frame_data["holder"]]
            if len(holder_data) == 1:
                temp = holder_data["id"].values[0]
            elif len(holder_data) >= 2:
                df.loc[
                    (df["frame"] == frame) & df["holder"] & df["prev_holder"],
                    "holder",
                ] = False
                # 再度holder_dataを取得（更新後）
                holder_data = df[(df["frame"] == frame) & df["holder"]]
                if len(holder_data) > 0:
                    temp = holder_data["id"].values[0]
            else:
                df.loc[(df["frame"] == frame) & (df["id"] == temp), "holder"] = True

        df = df.sort_values(by=["frame", "id"]).reset_index(drop=True)

        # Initialize the disc position
        df.loc[(df["class"] == "disc") & (df["frame"] != df["frame"].max()), "x"] = None
        df.loc[(df["class"] == "disc") & (df["frame"] != df["frame"].max()), "y"] = None

        # Get the frame where the holder has the disc
        holder_frame = df[df["holder"]]["frame"].unique()

        # Set the disc position to the holder's position
        for frame in holder_frame:
            ball_holder = df[df["holder"] & (df["frame"] == frame)]
            if len(ball_holder) > 0:
                df.loc[(df["class"] == "disc") & (df["frame"] == frame), "x"] = (
                    ball_holder["x"].values[0]
                )
                df.loc[(df["class"] == "disc") & (df["frame"] == frame), "y"] = (
                    ball_holder["y"].values[0]
                )

        # Interpolate the missing disc position
        df.loc[df["class"] == "disc", ["x", "y"]] = (
            df.loc[df["class"] == "disc", ["x", "y"]]
            .interpolate(method="linear", limit_direction="both")
            .round(2)
        )

        return df

    def generate_counterfactuals(self) -> Dict[str, CounterfactualResult]:
        """
        Generate counterfactual scenarios for all matches.

        Returns:
            Dictionary mapping match_id to CounterfactualResult.
        """
        input_dict = self.read_data()

        match_ids = sorted(input_dict.keys())
        if not match_ids:
            raise ValueError("No matches found in input data.")

        if self.testing_mode:
            print("Testing mode enabled: using only the first 3 matches.")
            match_ids = match_ids[:3]

        results: Dict[str, CounterfactualResult] = {}
        iterator: Iterable[str] = match_ids
        iterator = tqdm(match_ids, desc="Matches", leave=False)

        slide_sizes = range(*self.slide_size_range)

        for match_id in iterator:
            # Load DataFrame from file path
            input_path = input_dict[match_id]
            df = pd.read_csv(input_path)

            scenarios: Dict[int, pd.DataFrame] = {}

            for slide_size in slide_sizes:
                if slide_size < 0:
                    # Slide the movement data forward
                    df_slide = self._slide_movement_forward(df, slide_size)
                elif slide_size > 0:
                    # Slide the movement data backward
                    df_slide = self._slide_movement_backward(df, slide_size)
                else:
                    df_slide = df.copy()

                if df_slide is not None:
                    # Recalculate the disc position
                    df_slide = self._recalculate_disc_position(df_slide)
                    scenarios[slide_size] = df_slide

            result = CounterfactualResult(match_id, scenarios)
            results[match_id] = result

            if self.out_path:
                self._persist_result(match_id, scenarios)

        return results

    def _persist_result(
        self, match_id: str, scenarios: Dict[int, pd.DataFrame]
    ) -> None:
        """
        Persist counterfactual results to disk.

        Args:
            match_id: Match identifier
            scenarios: Dictionary mapping slide_size to DataFrame
        """
        assert self.out_path is not None
        data_dir = self.out_path
        data_dir.mkdir(parents=True, exist_ok=True)

        for slide_size, df in scenarios.items():
            output_file = data_dir / f"{match_id}_{slide_size}.csv"
            df.to_csv(output_file, index=False)
