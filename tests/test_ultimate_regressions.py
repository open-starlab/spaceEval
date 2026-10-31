"""Regressions for Ultimate's probability, identity, coordinate and fps contracts.

Load these modules directly so this test does not require unrelated sport extras.
Run: python -m unittest discover -s tests -p test_ultimate_regressions.py
"""
from pathlib import Path
import importlib.util
import inspect
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "spaceeval/sports/ultimate"
def _is_test_module(name):
    return name == "spaceeval" or name.startswith("spaceeval.") or name == "ultimate_numeric_regression"


def _load_modules():
    # These modules are loaded without unrelated sport dependencies. Restore
    # every affected sys.modules entry so other repository tests see exactly
    # the package/module identities that existed before this scoped load.
    previous = {name: module for name, module in sys.modules.items() if _is_test_module(name)}
    try:
        for name, path in [
            ("spaceeval", ROOT / "spaceeval"),
            ("spaceeval.sports", ROOT / "spaceeval/sports"),
            ("spaceeval.sports.ultimate", BASE),
            ("spaceeval.sports.ultimate.wuppcf", BASE / "wuppcf"),
            ("spaceeval.sports.ultimate.crsv", BASE / "crsv"),
        ]:
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = [str(path)]
                sys.modules[name] = module

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        return (
            load("spaceeval.sports.ultimate.wuppcf.config", BASE / "wuppcf/config.py"),
            load("ultimate_numeric_regression", BASE / "wuppcf/obso_repo/Metrica_PitchControl.py"),
            load("spaceeval.sports.ultimate.crsv.ultimate_crsv_main_class", BASE / "crsv/ultimate_crsv_main_class.py"),
            load("spaceeval.sports.ultimate.crsv.ultimate_generate_counterfactual_main_class", BASE / "crsv/ultimate_generate_counterfactual_main_class.py"),
            load("spaceeval.sports.ultimate.wuppcf.get_wuppcf_value", BASE / "wuppcf/get_wuppcf_value.py"),
        )
    finally:
        for name in list(sys.modules):
            if _is_test_module(name) and name not in previous:
                del sys.modules[name]
        sys.modules.update(previous)


config, mpc, crsv, cf, wuppcf = _load_modules()


def make_player(pid, side, xy, params):
    row = pd.Series({f"{side}_{pid}_x": xy[0], f"{side}_{pid}_y": xy[1],
                     f"{side}_{pid}_vx": 0., f"{side}_{pid}_vy": 0.,
                     "disc_x": -50., "disc_y": 0.})
    return mpc.player(pid, row, row, side, params)


def team_rows():
    home = {"disc_x": 0., "disc_y": 0.}
    away = home.copy()
    for i in range(1, 8):
        for row, side, xy in [(home, "Home", (0., 0.) if i == 1 else (float(i), 1.)),
                              (away, "Away", (2., 0.) if i == 1 else (float(i), 4.))]:
            row.update({f"{side}_{i}_x": xy[0], f"{side}_{i}_y": xy[1],
                        f"{side}_{i}_vx": 0., f"{side}_{i}_vy": 0.})
    return pd.Series(home), pd.Series(away)


class UltimateRegressions(unittest.TestCase):
    def setUp(self):
        self.params = config.get_model_params("UFATrack")

    def test_module_loading_preserves_existing_package_identity(self):
        before = {name: module for name, module in sys.modules.items() if _is_test_module(name)}
        _load_modules()
        after = {name: module for name, module in sys.modules.items() if _is_test_module(name)}
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            self.assertIs(after[name], before[name])
        sentinel = types.ModuleType("spaceeval")
        original = sys.modules.get("spaceeval")
        try:
            sys.modules["spaceeval"] = sentinel
            _load_modules()
            self.assertIs(sys.modules["spaceeval"], sentinel)
        finally:
            if original is None:
                sys.modules.pop("spaceeval", None)
            else:
                sys.modules["spaceeval"] = original

    def test_original_overshoot_is_bounded(self):
        p = self.params
        a = [make_player(i + 1, "Home", (-2. + i * .6, 0.), p) for i in range(6)]
        d = [make_player(i + 1, "Away", (-2. + i * .6, 2.), p) for i in range(7)]
        A, D, values = mpc.calculate_pitch_control_at_target(np.zeros(2), a, d, np.array([-50., 0.]), p)
        self.assertGreaterEqual(A, 0.)
        self.assertGreaterEqual(D, 0.)
        self.assertLessEqual(A + D, 1. + 1e-12)
        self.assertLessEqual(1. - A - D, p["model_converge_tol"])
        self.assertTrue(np.all(values >= 0.))
        self.assertAlmostEqual(values.sum(), A)

    def test_input_order_pruned_slots_and_repeated_calls(self):
        p = self.params
        a = [make_player(1, "Home", (50., 0.), p), make_player(2, "Home", (0., 0.), p)]
        d = [make_player(1, "Away", (0., 1.), p)]
        result = mpc.calculate_pitch_control_at_target(np.zeros(2), a, d, np.array([-1., 0.]), p)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[2].shape, (2,))
        self.assertEqual(result[2][0], 0.)
        self.assertGreater(result[2][1], 0.)
        again = mpc.calculate_pitch_control_at_target(np.zeros(2), a, d, np.array([-1., 0.]), p)
        np.testing.assert_array_equal(again[2], result[2])
        reversed_result = mpc.calculate_pitch_control_at_target(np.zeros(2), a[::-1], d, np.array([-1., 0.]), p)
        np.testing.assert_array_equal(reversed_result[2], result[2][::-1])
        self.assertEqual(list(inspect.signature(mpc.calculate_pitch_control_at_target).parameters),
                         ["target_position", "attacking_players", "defending_players", "ball_start_pos", "params"])
        for missing in (None, np.array([np.nan, np.nan])):
            self.assertTrue(np.isfinite(mpc.calculate_pitch_control_at_target(np.zeros(2), a, d, missing, p)[0]))

    def test_horizon_failure_and_explicit_retry(self):
        p = self.params
        a = [make_player(1, "Home", (50., 0.), p)]
        d = [make_player(1, "Away", (50., 1.), p)]
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            mpc.calculate_pitch_control_at_target(np.zeros(2), a, d, np.zeros(2), p)
        A, D, _ = mpc.calculate_pitch_control_at_target(np.zeros(2), a, d, np.zeros(2), dict(p, max_int_time=20.))
        self.assertLessEqual(1. - A - D, p["model_converge_tol"])

    def test_marker_distance_is_in_metres(self):
        home, away = team_rows()
        _, _, marker = mpc.initialise_players(home, away, "Home", "Away", self.params, 1)
        self.assertEqual(marker, 1)
        away["Away_1_x"] = 3.01
        _, _, marker = mpc.initialise_players(home, away, "Home", "Away", self.params, 1)
        self.assertIsNone(marker)

    def test_default_event_map_api_runs_without_removal(self):
        home, away = team_rows()
        event = pd.DataFrame([{"Start Frame": 0, "Start X": 0., "Start Y": 0.}])
        maps = mpc.generate_pitch_control_for_event(0, event, pd.DataFrame([home]), pd.DataFrame([away]),
            pd.DataFrame(), dict(self.params, max_int_time=20.), field_dimen=(10., 6.), n_grid_cells_x=5)
        self.assertEqual(len(maps), 4)
        self.assertIsNone(maps[2])
        self.assertEqual(maps[0].shape, (3, 5))
        self.assertEqual(maps[1].shape, (7, 3, 5))
        self.assertTrue(np.isfinite(maps[0]).all())

    def test_weight_grid_matches_metric_rows_and_raw_endpoints(self):
        ball = np.array([[0., 10.]])
        team, individual = mpc.calculate_ultimate_pitch_control(np.ones((1, 3, 5)),
            np.ones((1, 1, 3, 5)), ball, np.array([[np.nan, np.nan]]), field_dimen=(80., 20.))
        x, y = np.linspace(0., 80., 5), np.linspace(20., 0., 3)
        dist = np.hypot(x[None, :] - ball[0, 0], y[:, None] - ball[0, 1])
        sig = lambda a, b: 1. / (1. + np.exp(-a * (dist - b)))
        expected = (((2. - sig(.15, 40.)) / 2. + (2. - sig(.07, 40.)) / 2.) / 2. + 1. - sig(2., 70.)) / 2.
        np.testing.assert_allclose(team[0], expected)
        np.testing.assert_allclose(individual[0, 0], expected)
        # A noncentral disc detects an accidental independent vertical flip.
        other, _ = mpc.calculate_ultimate_pitch_control(np.ones((1, 3, 5)),
            np.ones((1, 1, 3, 5)), np.array([[0., 0.]]), np.array([[np.nan, np.nan]]), field_dimen=(80., 20.))
        self.assertGreater(other[0, -1, 0], other[0, 0, 0])

    def test_stall_and_range_weights_share_vertical_orientation(self):
        shape = (1, 3, 5)
        ball = np.array([[0., 0.]])
        no_marker, _ = mpc.calculate_ultimate_pitch_control(np.ones(shape), np.ones((1, *shape)),
            ball, np.array([[np.nan, np.nan]]), field_dimen=(12., 6.))
        marked, _ = mpc.calculate_ultimate_pitch_control(np.ones(shape), np.ones((1, *shape)),
            ball, np.array([[1., 0.]]), field_dimen=(12., 6.))
        self.assertLess(marked[0, -1, 2] / no_marker[0, -1, 2], .6)
        self.assertGreater(marked[0, 0, 2] / no_marker[0, 0, 2], .9)

    def test_crsv_grid_is_same_physical_grid(self):
        calc = crsv.ultimate_crsv(".", ".", provider="UFATrack")
        x, y = calc._get_grid_config()
        np.testing.assert_array_equal(x, np.linspace(0., 109.73, 54))
        np.testing.assert_array_equal(y, np.linspace(0., 48.77, 24))

    def test_changing_holder_keeps_global_player_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rows = []
            for frame, holder in [(0, 8), (1, 9)]:
                for pid in [8, 9, 10]:
                    rows.append(dict(frame=frame, id=pid, **{"class": "offense"}, holder=pid == holder,
                                     selected=pid == 10, x=30., y=20., vx=5., vy=0.))
                rows.append(dict(frame=frame, id=15, **{"class": "disc"}, holder=False,
                                 selected=False, x=0., y=20., vx=0., vy=0.))
            pd.DataFrame(rows).to_csv(path / "example_0.csv", index=False)
            # The exporter sorts string IDs: "10", "8", "9".
            maps = np.broadcast_to(np.array([.3, .1, .2])[:, None, None, None], (3, 2, 24, 54)).copy()
            np.save(path / "example_0.npy", maps)
            out = crsv.ultimate_crsv(path, path, provider="UFATrack").calc_v_frame()["example_0"].v_frame_data
            np.testing.assert_allclose(out["value"], [.3, .3])
            np.save(path / "example_0.npy", maps[:2])
            with self.assertRaisesRegex(ValueError, "Player-map axis"):
                crsv.ultimate_crsv(path, path, provider="UFATrack").calc_v_frame()

    def test_exported_player_maps_keep_identity_in_crsv(self):
        # Exercise the actual CSV reader, exporter axis assembly, saved .npy,
        # and CRSV consumer. Tagged maps isolate player identity from physics.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            scenarios = path / "scenarios"
            scenarios.mkdir()
            holders = [8, 9]
            events = pd.DataFrame({"Start Frame": [0, 1], "From": holders,
                                   "Start X": [0., 0.], "Start Y": [20., 20.]})
            tracking = pd.DataFrame({"Frame": [0, 1], "Time [s]": [0., .1]})
            events.to_csv(path / "events.csv", index=False)
            tracking.to_csv(path / "tracking.csv", index=False)
            rows = []
            for frame, holder in enumerate(holders):
                for pid in [8, 9, 10]:
                    rows.append(dict(frame=frame, id=pid, **{"class": "offense"}, holder=pid == holder,
                                     selected=pid == 10, x=30., y=20., vx=5., vy=0.))
                rows.append(dict(frame=frame, id=15, **{"class": "disc"}, holder=False,
                                 selected=False, x=0., y=20., vx=0., vy=0.))
            pd.DataFrame(rows).to_csv(scenarios / "example_0.csv", index=False)

            def tagged_event_map(event_idx, *args, **kwargs):
                ids = [pid for pid in [8, 9, 10] if pid != holders[event_idx]]
                maps = np.stack([np.full((24, 54), pid / 100.) for pid in ids])
                return np.ones((24, 54)), maps, None, [types.SimpleNamespace(id=str(pid)) for pid in ids]

            with patch.object(wuppcf.mvel, "calc_player_velocities", side_effect=lambda data, **kw: data), \
                 patch.object(wuppcf.mpc, "generate_pitch_control_for_event", side_effect=tagged_event_map), \
                 patch.object(wuppcf.mpc, "calculate_ultimate_pitch_control", side_effect=lambda team, players, *args, **kw: (team, players)):
                exported = wuppcf.calculate_wuppcf(str(path / "events.csv"), str(path / "tracking.csv"),
                                                  str(path / "tracking.csv"), provider="UFATrack")
            np.testing.assert_allclose(exported.player_wuppcf[:, :, 0, 0],
                                       [[.10, .10], [0., .08], [.09, 0.]])
            np.save(path / "example_0.npy", exported.player_wuppcf)
            result = crsv.ultimate_crsv(path, scenarios, provider="UFATrack").calc_v_frame()["example_0"]
            np.testing.assert_allclose(result.v_frame_data["value"], [.10, .10])

    def test_disk_scenario_ids_and_short_smoothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); (path / "v_frame").mkdir()
            values = np.arange(9.) / 10.
            pd.DataFrame({"frame": np.arange(9), "value": values}).to_csv(path / "v_frame/example_0.csv", index=False)
            calc = crsv.ultimate_crsv(path, path, out_path=path)
            out = calc.calc_v_scenario()["example"].v_scenario_data
            expected = np.convolve(values, np.ones(10), mode="full")[4:13] / 10.
            self.assertEqual(out.iloc[0]["shift"], 0)
            self.assertEqual(out.iloc[0]["v_scenario"], np.max(expected).round(3))
            long_values = np.arange(20.) / 20.
            result = crsv.VFrameResult("long_0", pd.DataFrame({"value": long_values}))
            calc.calc_v_scenario({"long_0": result})
            np.testing.assert_array_equal(result.v_frame_data["value_smoothed"],
                                          np.convolve(long_values, np.ones(10), mode="same") / 10.)

    def test_counterfactual_displacement_uses_provider_fps(self):
        columns = "frame,id,x,y,vx,vy,ax,ay,v_mag,a_mag,v_angle,a_angle,diff_v_a_angle,diff_v_angle,diff_a_angle,class,holder,closest,selected,prev_holder,def_selected".split(",")
        rows = []
        for frame in range(30):
            for pid, kind in [(1, "offense"), (8, "defense")]:
                row = dict.fromkeys(columns, 0.)
                row.update(frame=frame, id=pid, x=frame, y=pid, vx=10., vy=0.,
                           **{"class": kind}, holder=False, closest=8, selected=pid == 1 and 10 <= frame <= 15,
                           prev_holder=False, def_selected=pid == 8 and 10 <= frame <= 15)
                rows.append(row)
        data = pd.DataFrame(rows, columns=columns)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "example.csv"
            data.to_csv(source, index=False)
            generator = cf.ultimate_generate_counterfactual(input_data=source, provider="UFATrack", slide_size_range=(-1, 2, 1))
            scenarios = generator.generate_counterfactuals()["example"].scenarios
        forward, backward = scenarios[-1], scenarios[1]
        for pid in [1, 8]:
            tail = forward.loc[forward.id == pid].set_index("frame")
            self.assertAlmostEqual(tail.loc[29, "x"] - tail.loc[28, "x"], 1.)
            bridge = backward.loc[backward.id == pid].set_index("frame")
            self.assertAlmostEqual(bridge.loc[10, "x"] - bridge.loc[9, "x"], 1.)


if __name__ == "__main__":
    unittest.main()
