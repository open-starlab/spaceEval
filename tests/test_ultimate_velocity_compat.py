"""SciPy compatibility for Ultimate velocity smoothing and missing samples."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SOURCE = Path(__file__).resolve().parents[1] / "spaceeval/sports/ultimate/wuppcf/obso_repo/Metrica_Velocities.py"
spec = importlib.util.spec_from_file_location("ultimate_velocity_compat", SOURCE)
velocities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(velocities)


class UltimateVelocityCompatibility(unittest.TestCase):
    def test_finite_series_matches_scipy_including_edges(self):
        values = np.array([0., 1., 4., 2., 6., 3., 8., 2., 5., 7.])
        for window in (1, 4, 5):
            for order in range(min(window, 3)):
                with self.subTest(window=window, order=order):
                    np.testing.assert_allclose(velocities._smooth_velocity(values, window, order),
                                               savgol_filter(values, window, order), atol=1e-13)

    def test_missing_samples_stay_missing_in_affected_windows(self):
        for missing, affected in ((0, slice(0, 3)), (7, slice(5, 10)), (14, slice(12, 15))):
            with self.subTest(missing=missing):
                values = np.ones(15)
                values[missing] = np.nan
                expected = np.ones(15)
                expected[affected] = np.nan
                np.testing.assert_allclose(velocities._smooth_velocity(values, 5, 0), expected,
                                           equal_nan=True)

    def test_stationary_tracking_keeps_first_difference_missing(self):
        data = pd.DataFrame({"Time [s]": np.arange(10) / 10.,
                             "Home_1_x": np.ones(10), "Home_1_y": np.ones(10)})
        result = velocities.calc_player_velocities(data)
        for column in ("Home_1_vx", "Home_1_vy"):
            self.assertTrue(result[column].iloc[:3].isna().all())
            np.testing.assert_array_equal(result[column].iloc[3:], 0.)

    def test_short_input_still_raises(self):
        with self.assertRaises(ValueError):
            velocities._smooth_velocity(np.ones(2), 5, 0)


if __name__ == "__main__":
    unittest.main()
