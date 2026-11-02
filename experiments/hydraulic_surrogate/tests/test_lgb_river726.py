"""Guard prospective river-model eligibility, target order and independent locks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import train_lgb_river726 as runner


class River726Tests(unittest.TestCase):
    def test_raw_control_requires_its_own_independent_checkpoint_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'artifacts/lgb-shop664-2021-001'
            directory.mkdir(parents=True)
            prediction = directory / 'validation_2021.npz'
            prediction.write_bytes(b'opaque raw control')
            manifest = directory / 'manifest.json'
            manifest.write_text(json.dumps({'status': 'complete', 'protocol_sha256': runner.BASE_SHA,
                'artifacts': {'validation_2021.npz': runner.base.sha(prediction)}}))
            audit_path = root / 'artifacts/lgb-shop664-2021-independent-audit-001/manifest.json'
            audit_path.parent.mkdir(parents=True)
            audit = {'status': 'passed', 'protocol_sha256': runner.BASE_SHA,
                'candidate_manifest_sha256': runner.base.sha(manifest),
                'all14_native_checkpoints_replayed': True, 'original_target_values_verified': True}
            record = {}
            for key, path in [('prediction', prediction), ('manifest', manifest), ('audit', audit_path)]:
                record[key] = path.relative_to(root).as_posix()
                if key != 'audit':
                    record[key + '_sha256'] = runner.base.sha(path)
            for change in [{}, {'candidate_manifest_sha256': 'foreign'},
                           {'all14_native_checkpoints_replayed': False}, {'status': 'failed'}]:
                audit_path.write_text(json.dumps({**audit, **change}))
                record['audit_sha256'] = runner.base.sha(audit_path)
                with patch.object(runner, 'ROOT', root):
                    if change:
                        with self.assertRaises(AssertionError):
                            runner.check_control(record, '2021')
                    else:
                        runner.check_control(record, '2021')

    def test_lock_requires_eligible_same_protocol_and_independent_review(self):
        p = {'source': {'scripts/train_lgb_river726.py': 'trainer'},
             'anchor': {'2021': {'prediction_sha256': 'anchor'}},
             'control': {'2021': {'prediction_sha256': 'control'}}}
        lock = {'version': runner.LOCK_VERSION, 'selection_year': 2021,
                'protocol_sha256': 'protocol', 'trainer_sha256': 'trainer',
                'candidate_weights': runner.base.WEIGHTS,
                'scores_2021': {'0.0': .98, '0.1': .99, '0.25': .995, '0.5': .99},
                'selected_weight': .25, 'selected_2021_micro_auc': .995,
                'anchor_2021_micro_auc': .98, 'control_2021_micro_auc': .994,
                'eligible_for_2022': True, 'anchor_prediction_sha256': 'anchor',
                'control_prediction_sha256': 'control'}
        with tempfile.TemporaryDirectory() as tmp:
            lp, ap = Path(tmp) / 'lock.json', Path(tmp) / 'audit.json'
            for change in [{}, {'selected_weight': .1}, {'eligible_for_2022': False},
                           {'protocol_sha256': 'wrong'}, {'control_prediction_sha256': 'wrong'},
                           {'version': 'lgb_shop664_selection_lock_v1'}, {'selection_year': 2022}]:
                lp.write_text(json.dumps({**lock, **change}))
                ap.write_text(json.dumps({'status': 'passed', 'selection_lock_sha256': runner.base.sha(lp),
                                         'protocol_sha256': 'protocol', 'eligible_for_2022': True}))
                args = lp, runner.base.sha(lp), p, 'protocol', ap, runner.base.sha(ap)
                if change:
                    with self.assertRaises(AssertionError):
                        runner.validate_lock(*args)
                else:
                    self.assertEqual(runner.validate_lock(*args)['selected_weight'], .25)
            lp.write_text(json.dumps(lock))
            ap.write_text(json.dumps({'status': 'passed', 'selection_lock_sha256': runner.base.sha(lp),
                                     'protocol_sha256': 'foreign', 'eligible_for_2022': True}))
            with self.assertRaises(AssertionError):
                runner.validate_lock(lp, runner.base.sha(lp), p, 'protocol', ap, runner.base.sha(ap))

    def test_partial_cache_rejected_before_features_or_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'cache').mkdir()
            mp, pp, ap = root / 'cache/manifest.json', root / 'cache/protocol.json', root / 'audit.json'
            mp.write_text(json.dumps({'status': 'incomplete'}))
            pp.write_text('{}')
            ap.write_text('{}')
            record = {'directory': 'cache', 'manifest_sha256': runner.base.sha(mp),
                      'protocol_sha256': runner.base.sha(pp), 'audit': 'audit.json',
                      'audit_sha256': runner.base.sha(ap)}
            with patch.object(runner, 'ROOT', root):
                with self.assertRaises(AssertionError):
                    runner.check_cache_contract(record)

    def test_cache_must_have_full_independent_physical_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'cache').mkdir()
            mp, pp, ap = root / 'cache/manifest.json', root / 'cache/protocol.json', root / 'audit.json'
            case_ids = list(range(1, 3648))
            pp.write_text(json.dumps({'case_ids': case_ids}))
            m = {'status': 'complete', 'failures': [], 'protocol_sha256': runner.base.sha(pp),
                 'case_ids': case_ids, 'completed_cases': {str(i): {} for i in case_ids},
                 'all_cases_physical_and_replay_verified': True}
            mp.write_text(json.dumps(m))
            audit = {'status': 'passed', 'case_count': 3647, 'cache_manifest_sha256': runner.base.sha(mp),
                     'cache_protocol_sha256': runner.base.sha(pp), 'all_cases_physical_and_replay_verified': False,
                     'canonical_reuse_verified': True}
            ap.write_text(json.dumps(audit))
            record = {'directory': 'cache', 'manifest_sha256': runner.base.sha(mp),
                      'protocol_sha256': runner.base.sha(pp), 'audit': 'audit.json',
                      'audit_sha256': runner.base.sha(ap)}
            with patch.object(runner, 'ROOT', root):
                with self.assertRaises(AssertionError):
                    runner.check_cache_contract(record)

    def test_identity_ties_and_same_weight_control_are_preserved(self):
        self.assertEqual(runner.base.choose({str(w): .99 for w in runner.base.WEIGHTS}, .98), (0., False))
        scores = {'0.0': .98, '0.1': .99, '0.25': .99, '0.5': .98}
        self.assertEqual(runner.base.choose(scores, .99), (.1, False))
        self.assertEqual(runner.base.choose(scores, .989), (.1, True))
        self.assertEqual(runner.base.choose({'0.0': .98, '0.25': .99}, .989, .25), (.25, True))


if __name__ == '__main__':
    unittest.main()
