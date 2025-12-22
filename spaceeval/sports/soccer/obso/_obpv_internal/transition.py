import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import os
from tqdm import tqdm

from ..c_obso_repo import Metrica_IO as mio


class TransitionKernelModel:

    def __init__(self, event_data, tracking_home, tracking_away):
        self.event_data = event_data
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.pass_direction_matrices: dict[int, list[list[float]]] = {}
        self.transition_distributions: dict[int, pd.DataFrame]
        self.settings = {
            'field_dimen': (100., 64.),
        }

    def fit(self):
        directions = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [
        ], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: []}

        iterator = tqdm(self.event_data.keys(),
                        desc="Fitting Transition Matrices")
        for idx in iterator:
            events = self.event_data[idx]
            tracking_home = self.tracking_home[idx]
            tracking_away = self.tracking_away[idx]

            tracking_home, tracking_away, events = mio.to_single_playing_direction(
                tracking_home, tracking_away, events)

            for idx, event in events.iterrows():
                if pd.isna(event['Type']) or event['Type'].lower() != 'pass':
                    continue

                team = event['Team']
                start_x = event['Start X']
                start_y = event['Start Y']
                end_x = event['End X']
                end_y = event['End Y']

                if team == 'Away':
                    start_x = -start_x
                    start_y = -start_y
                    end_x = -end_x
                    end_y = -end_y

                area_number = self.divide_pitch(
                    start_x, start_y, dimensions=self.settings['field_dimen'])

                directions[area_number].append(
                    [end_x - start_x, end_y - start_y])

        for key in directions.keys():
            if len(directions[key]) == 0:
                directions[key].append([0.0, 0.0])
        self.pass_direction_matrices = directions
        self.transition_distributions = self._kde_distribution(
            self.pass_direction_matrices)
        return self.transition_distributions

    def predict(self, ball_x: float, ball_y: float):
        if not self.transition_distributions:
            raise ValueError(
                "Transition distributions not fitted yet. Call 'fit' method first.")

        area_number = self.divide_pitch(ball_x, ball_y)
        transition_matrix = self.transition_distributions.get(area_number)
        return transition_matrix

    def save_to_csv(self, save_dir):
        pass_head = ['x', 'y']
        os.makedirs(save_dir, exist_ok=True)
        for key in self.pass_direction_matrices.keys():
            pd.DataFrame(self.pass_direction_matrices[key]).to_csv(
                f'{save_dir}/Area{key}_pass.csv', index=False, header=pass_head)

            self.transition_distributions[key].to_csv(
                f'{save_dir}/Area{key}_Transition.csv', index=False, header=False)
    
    def load_from_csv(self, load_dir):
        self.pass_direction_matrices = {}
        self.transition_distributions = {}
        for file_name in os.listdir(load_dir):
            if file_name.startswith("Area") and file_name.endswith("_pass.csv"):
                area_number = int(file_name[len("Area"): -len("_pass.csv")])
                df = pd.read_csv(os.path.join(load_dir, file_name))
                self.pass_direction_matrices[area_number] = df.values.tolist()
            elif file_name.startswith("Area") and file_name.endswith("_Transition.csv"):
                area_number = int(
                    file_name[len("Area"): -len("_Transition.csv")])
                df = pd.read_csv(os.path.join(load_dir, file_name), header=None)
                self.transition_distributions[area_number] = df

    @staticmethod
    def divide_pitch(x, y, dimensions=(106., 68.)):
        """
        Returns the area number based on the given x and y coordinates on the pitch.

            Pitch layout (in meters):
                    attacking direction ----->
                    x_pitch = 105
            -------------------------------------------------   |
            |   1   |   4   |   7   |  10   |  13   |  16   |   |y_pitch = 68
            -------------------------------------------------   |
           [|   2   |   5   |   8   |  11   |  14   |  17   |]  |
            -------------------------------------------------   |
            |   3   |   6   |   9   |  12   |  15   |  18   |   |
            -------------------------------------------------   v

            The way the pitch is divided is based on coach Van Haar's idea.

        Parameters
        ----------
        x : float
            x coordinate on the pitch (in meters).
        y : float
            y coordinate on the pitch (in meters).

        Returns
        -------
        int
            Area number on the pitch.

        """

        length, width = dimensions
        base_x_scale = 106.0 / length
        base_y_scale = 68.0 / width
        x = x * base_x_scale
        y = y * base_y_scale

        if x < -36:
            x_value = 0
        elif x < -18:
            x_value = 1
        elif x < 0:
            x_value = 2
        elif x < 18:
            x_value = 3
        elif x < 36:
            x_value = 4
        else:
            x_value = 5

        if y < -14:
            y_value = 1
        elif y < 14:
            y_value = 2
        else:
            y_value = 3

        pitch_number = 3 * x_value + y_value

        return pitch_number

    def visualize_passmap():
        pass

    def _kde_distribution(self, directions, field_dimen=(106., 68.), n_grid_cells_x=50, bandwidth_factor=2.0):

        n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
        dx = field_dimen[0] / n_grid_cells_x
        dy = field_dimen[1] / n_grid_cells_y

        x_grid = np.linspace(-field_dimen[0], field_dimen[0], n_grid_cells_x*2)
        y_grid = np.linspace(-field_dimen[1], field_dimen[1], n_grid_cells_y*2)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        results = {}

        for key in directions.keys():
            x_coords = np.array(directions[key])[:, 0]
            y_coords = np.array(directions[key])[:, 1]

            if key % 3 == 1:
                x_coords = np.concatenate(
                    [x_coords, np.array(directions[key+2])[:, 0]])
                y_coords = np.concatenate(
                    [y_coords, -np.array(directions[key+2])[:, 1]])

            elif key % 3 == 2:
                x_coords = np.concatenate([x_coords, x_coords])
                y_coords = np.concatenate([y_coords, -y_coords])

            elif key % 3 == 0:
                x_coords = np.concatenate(
                    [x_coords, np.array(directions[key-2])[:, 0]])
                y_coords = np.concatenate(
                    [y_coords, -np.array(directions[key-2])[:, 1]])

            xy = np.vstack([x_coords, y_coords])

            if xy.shape[1] < 2:
                print(
                    f"Not enough samples to fit KDE for area {key}. Storing zeros.")
                df = pd.DataFrame(np.zeros(X.shape))
                results[key] = df
                continue

            # if contains infinite values or NaN, remove them
            xy = xy[:, np.isfinite(xy).all(axis=0)]

            try:
                kde = gaussian_kde(xy, bw_method='silverman')
            except np.linalg.LinAlgError:
                print(
                    f"Singular covariance matrix encountered for area {key}. Adding jitter and retrying.")
                # singular covariance: try adding tiny jitter and retry
                jitter_scale = 1e-6 * (np.nanmax(np.std(xy, axis=1)) + 1e-8)
                rng = np.random.default_rng(0)
                xy_jitter = xy + rng.normal(scale=jitter_scale, size=xy.shape)
                try:
                    kde = gaussian_kde(xy_jitter, bw_method='silverman')
                except np.linalg.LinAlgError:
                    print(
                        f"Failed to fit KDE for area {key} after adding jitter. Storing zeros.")
                    df = pd.DataFrame(np.zeros(X.shape))
                    results[key] = df
                    continue

            kde.set_bandwidth(bw_method=kde.factor * bandwidth_factor)

            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kde(positions).T, X.shape)

            df = pd.DataFrame(Z)
            results[key] = df
        return results
