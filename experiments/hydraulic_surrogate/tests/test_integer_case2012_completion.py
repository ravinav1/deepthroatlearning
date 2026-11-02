"""No-solve rejection tests for the narrowly declared numerical completion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from scripts.complete_integer_case2012 import (
    ROOT, ORIGINAL_CASE, PAYLOADS, dimensions, digest, failed_record, verify_fresh,
)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / ORIGINAL_CASE / 'replica_1_60s/proof.json').read_text())

    def test_only_exact_case_native_status_and_failure_are_eligible(self):
        failed_record(self.record)
        for field, value in [('case_id', 2013), ('status', 'complete'), ('milp_limit_seconds', 600),
                             ('error', 'some other numerical failure')]:
            record = copy.deepcopy(self.record)
            record[field] = value
            with self.assertRaises(ValueError):
                failed_record(record)
        record = copy.deepcopy(self.record)
        record['proof']['mip_status'] = 1
        with self.assertRaises(ValueError):
            failed_record(record)

    def test_certificate_rejects_bad_scale_or_truncated_duals(self):
        problem = {'c': np.ones(2), 'bounds': np.zeros((2, 2)),
                   'A_eq': csr_matrix((1, 2)), 'b_eq': np.zeros(1),
                   'A_ub': csr_matrix((3, 2)), 'b_ub': np.zeros(3)}
        native = {'integer_solution': np.zeros(2), 'integer_states': np.zeros((3, 168))}
        lp = {'solution': np.zeros(2)}
        for kind, count in [('eqlin', 1), ('ineqlin', 3), ('lower', 2), ('upper', 2)]:
            for suffix in ['residual', 'marginals']:
                lp[kind + '_' + suffix] = np.zeros(count)
        dimensions(problem, native, lp, 1.)
        for scale in [0., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                dimensions(problem, native, lp, scale)
        lp['ineqlin_marginals'] = np.zeros(2)
        with self.assertRaises(ValueError):
            dimensions(problem, native, lp, 1.)

    def test_self_consistent_new_payload_still_must_match_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            folders = [base / name for name in ['old1', 'old2', 'fresh']]
            records = []
            for folder in folders:
                folder.mkdir()
                record = copy.deepcopy(self.record)
                for name in PAYLOADS:
                    np.savez_compressed(folder / name, value=np.asarray([1., 2.]))
                record['artifacts'] = {name: digest(folder / name) for name in PAYLOADS}
                (folder / 'proof.json').write_text(json.dumps(record))
                records.append(record)
            p = {'environment': self.record['environment'],
                 'case_input_sha256': self.record['input_sha256'],
                 'case_control_sha256': self.record['control_sha256'],
                 'original_attempts': [{'directory': str(folder), 'payloads': record['artifacts']}
                                       for folder, record in zip(folders[:2], records[:2])]}
            verify_fresh(folders[2], records[2], p)
            np.savez_compressed(folders[2] / PAYLOADS[0], value=np.asarray([1., 2.00001]))
            records[2]['artifacts'][PAYLOADS[0]] = digest(folders[2] / PAYLOADS[0])
            with self.assertRaises(ValueError):
                verify_fresh(folders[2], records[2], p)

    def test_altered_native_bound_is_rejected_even_with_identical_payloads(self):
        original = ROOT / ORIGINAL_CASE / 'replica_1_60s'
        p = {'environment': self.record['environment'],
             'case_input_sha256': self.record['input_sha256'],
             'case_control_sha256': self.record['control_sha256'],
             'original_attempts': [{'directory': str(original), 'payloads': self.record['artifacts']}]}
        record = copy.deepcopy(self.record)
        record['proof']['mip_dual_bound_scaled'] += 1.
        with self.assertRaises(ValueError):
            verify_fresh(original, record, p)


if __name__ == '__main__':
    unittest.main()
