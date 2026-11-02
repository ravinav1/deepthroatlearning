"""Independent actual-screen auditor contracts using synthetic inputs only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts import audit_canonical_pooled_result as audit
from sintef_project.canonical_unit_features import canonical_block, feature_specification


class CanonicalResultAuditTest(unittest.TestCase):
    def test_independent_semantic_mapping_and_signed_zero(self):
        p = json.loads((audit.ROOT / 'docs/lgb_shop664_protocol.json').read_text())
        rng = np.random.default_rng(372)
        frame = pd.DataFrame(rng.normal(size=(7, 664)), columns=p['feature_columns'])
        frame.iloc[3] = -0.0
        rows = np.array([6, 3, 0, 4], dtype=np.int64)
        independent = audit.independent_block(frame, rows, p['units'])
        expected = canonical_block(frame, rows, feature_specification(p['feature_columns'], p['units']))
        self.assertEqual(independent.shape, (56, 115))
        self.assertEqual(independent.dtype, np.float32)
        self.assertEqual(independent.tobytes(), expected.tobytes())
        self.assertTrue(np.signbit(independent[14:28, :111]).all())
        self.assertEqual(independent[2 * 14 + 9, 111], 9)

    def test_native_recipe_and_full_synthetic_checkpoint_replay(self):
        directory = audit.ROOT / 'artifacts/canonical-pooled-timing-001'
        model = audit.native_model(directory / 'pooled.txt', 64, 64)
        x = np.load(directory / 'pooled_x.npy', mmap_mode='r', allow_pickle=False)
        y = np.load(directory / 'pooled_prediction.npy', allow_pickle=False).ravel()
        audit.replay(model, x, y)
        altered = y.copy()
        altered[-1] = np.nextafter(altered[-1], np.inf)
        with self.assertRaises(AssertionError):
            audit.replay(model, x, altered)
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / 'changed.txt'
            bad.write_text((directory / 'pooled.txt').read_text().replace('[categorical_feature: 111,112]', '[categorical_feature: 111]'))
            with self.assertRaises(AssertionError):
                audit.native_model(bad, 64, 64)

    def test_independent_decision_ties_and_failed_selected_half_no_rescue(self):
        anchor = dict.fromkeys(['full', 'first_half', 'second_half'], .8)
        controls = {str(w): dict(anchor) for w in audit.WEIGHTS}
        grid = {f'{r}:{w}': dict(anchor) for r in audit.ROUNDS for w in audit.WEIGHTS}
        for r in audit.ROUNDS:
            for w in [.1, .25]:
                grid[f'{r}:{w}'] = dict.fromkeys(anchor, .9)
        result = audit.decide(grid, controls, anchor)
        self.assertEqual((result['selected_rounds'], result['selected_weight']), (700, .1))
        self.assertTrue(result['eligible_for_later_validation'])
        grid['1400:0.5'] = {'full': .95, 'first_half': .96, 'second_half': .79}
        result = audit.decide(grid, controls, anchor)
        self.assertEqual((result['selected_rounds'], result['selected_weight']), (1400, .5))
        self.assertFalse(result['eligible_for_later_validation'])

    def test_exit_evidence_wrong_candidate_live_or_early_rejected(self):
        m = {'protocol_sha256': 'p', 'command': 'python screen', 'timestamp_utc': '2026-11-02T20:00:00+00:00', 'runtime_seconds': 30}
        o = {'status': 'passed', 'version': 'canonical_pooled_process_exit_v1', 'candidate_manifest_sha256': 'm',
             'protocol_sha256': 'p', 'observed_launch_command': 'python screen', 'launch_observed': True,
             'producer_pid': 10, 'process_exited': True, 'command': ['ps', '-p', '10', '-o', 'pid=,stat=,command='],
             'returncode': 1, 'stdout': '', 'stderr': '', 'timestamp_utc': '2026-11-02T20:00:31+00:00'}
        audit.validate_exit(o, m, 'm')
        for key, value in [('candidate_manifest_sha256', 'other'), ('process_exited', False), ('stdout', '10 R python'),
                           ('timestamp_utc', '2026-11-02T20:00:29+00:00')]:
            with self.assertRaises(AssertionError):
                audit.validate_exit({**o, key: value}, m, 'm')

    def test_target_reader_exact_selected_rows_and_unit_hour_order(self):
        metadata = pd.DataFrame({'Run No': [1, 2, 3, 4]})
        units = [f'U{n}' for n in range(14)]
        cases = np.array([0, 3])
        def reader(path, **kwargs):
            self.assertEqual(kwargs['skiprows'], [2, 3])
            values = {'Run No': [1, 4]}
            values.update({f'result_committed_{u}_t{h}': np.array([(i+h)%2, (i+h+1)%2], dtype=np.uint8)
                           for i, u in enumerate(units) for h in range(168)})
            return pd.DataFrame(values)
        with patch.object(audit.pd, 'read_csv', side_effect=reader):
            values = audit.read_targets(metadata, cases, units)
        self.assertEqual(values.shape, (2, 168, 14))
        self.assertEqual(values[1, 167, 8], (8+167+1)%2)

    def test_path_alias_and_metric_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'value').write_text('x')
            (base / 'alias').symlink_to(base / 'value')
            self.assertEqual(audit.safe(base, 'value'), base / 'value')
            for name in ['../escape', 'alias', '/absolute']:
                with self.assertRaises(AssertionError): audit.safe(base, name)
        audit.compare_scores({'x': {'full': .9}}, {'x': {'full': .9+1e-15}})
        with self.assertRaises(AssertionError):
            audit.compare_scores({'x': {'full': .9}}, {'x': {'full': .9001}})


if __name__ == '__main__':
    unittest.main()
