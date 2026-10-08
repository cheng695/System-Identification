"""CPU smoke/regression tests; all generated results stay in temporary directories."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from gimbal_sysid import config, train
from gimbal_sysid.data import TrajectoryData
from gimbal_sysid.results import load_result, generate_report
from gimbal_sysid.visualize import load_saved_trajectory


def trajectory(n):
    return TrajectoryData(
        np.zeros((n, 1), np.float32), np.zeros((n, 1), np.float32),
        np.full((n, 1), 0.01, np.float32),
    ).with_model_configuration(['joint'], [0], [], [])


class WindowTests(unittest.TestCase):
    def test_coverage_and_holdout(self):
        with patch.object(config, 'TRAINING_WINDOW_STEPS', 3):
            for n in (12, 13, 14):
                data = trajectory(n)
                # Low activity must not remove evaluation samples.
                data.pos_real[:, 0] = np.arange(n)
                data.tor_real[:] = 0
                windows = train.make_training_windows(data, limit_count=False)
                np.testing.assert_array_equal(np.concatenate([w.pos_real for w in windows]), data.pos_real)
                fit, validation = train.split_window_ids(windows)
                self.assertFalse(set(fit) & set(validation))
                self.assertEqual(set(fit) | set(validation), {i for i, w in enumerate(windows) if w.num_steps > 1})
                batches = train.iter_window_batches(fit, 2, 42)
                for cycle in (1, 2):
                    visited = []
                    for _ in range(int(np.ceil(len(fit) / 2))):
                        actual_cycle, ids = next(batches)
                        self.assertEqual(cycle, actual_cycle)
                        visited.extend(ids)
                    self.assertEqual(sorted(visited), sorted(fit))

    def test_real_mjx_training_and_output(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
            <joint name="joint"/><geom type="sphere" size="0.1" mass="1"/>
            </body></worldbody><actuator><motor joint="joint"/></actuator></mujoco>''')
        mx = mjx.put_model(model)
        data = trajectory(13)
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(config, RESULTS_DIR=Path(directory), NUM_EPOCHS=3,
                                TRAINING_WINDOW_STEPS=3, TRAINING_WINDOW_COUNT=2,
                                VALIDATION_EVERY=2, SAVE_FREE_ROLLOUT=False), \
                 patch.object(train, 'choose_identification_mode', return_value='all'), \
                 patch.object(train, 'load_experiment_interactively', return_value=('test.xml', data)), \
                 patch.object(train, 'ask_sample_dt', return_value=.002), \
                 patch.object(train, 'load_model', return_value=(model, mx)):
                train.main()
                path = Path(directory) / config.RESULT_FILENAME
                result = load_result(path)
                self.assertEqual(result['simulation_trajectory'].shape, (13, 2))
                self.assertEqual(str(result['parameter_selection']), 'best_validation_loss')
                self.assertEqual(set(result['visited_training_window_ids']), set(result['training_window_ids']))
                self.assertFalse(set(result['training_window_ids']) & set(result['validation_window_ids']))
                self.assertAlmostEqual(float(result['best_loss']), result['validation_history'][:, 1].min())
                np.testing.assert_array_equal(result['evaluation_pos_real'], data.pos_real)
                load_saved_trajectory(path)
                generate_report(path)
                self.assertTrue((path.parent / 'figures/segment_comparison_0.png').exists())
                # Finite gradient and finite-difference agreement for the dynamic objective.
                raw = {'viscous': jnp.log(jnp.array([.005])), 'coulomb': jnp.log(jnp.array([.005]))}
                objective = jax.jit(train.make_objective(model, mx, data, 'friction'))
                args = (data.pos_real[:3], data.vel_real[:3], data.tor_real[:3])
                gradient = jax.grad(objective)(raw, *args)
                for key in raw:
                    plus, minus = dict(raw), dict(raw)
                    plus[key], minus[key] = raw[key] + .01, raw[key] - .01
                    finite_difference = (float(objective(plus, *args)) - float(objective(minus, *args))) / .02
                    np.testing.assert_allclose(float(gradient[key][0]), finite_difference, rtol=.1, atol=1e-7)
                original = path.read_bytes()
                with patch.object(train, 'make_evaluation_rollout', return_value=lambda *args: np.full((3, 2), np.nan)):
                    with self.assertRaises(RuntimeError):
                        train.main()
                self.assertEqual(original, path.read_bytes())
                self.assertTrue(path.with_name(path.stem + '_checkpoint.npz').exists())


if __name__ == '__main__':
    unittest.main()
