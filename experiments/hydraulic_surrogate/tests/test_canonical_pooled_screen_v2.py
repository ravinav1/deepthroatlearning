"""Execution-gate ordering and unchanged canonical selection behavior."""
import ast
import inspect
import unittest
from unittest.mock import patch

from scripts import screen_canonical_pooled_v2 as candidate
from tests.test_canonical_pooled_screen import CanonicalPooledScreenTest


class CanonicalPooledV2Test(unittest.TestCase):
    def test_observed_resource_failure_rejects(self):
        protocol = {'resources': {'entry_free_percent': 40., 'ongoing_free_percent': 20.,
                                 'observed_peak_limit_bytes': 1000, 'observed_wall_limit_seconds': 30}}
        with patch.object(candidate.base, 'memory_free_percent', return_value=70.), \
             patch.object(candidate.base, 'peak_rss', return_value=500), \
             patch.object(candidate.time, 'monotonic', return_value=40.):
            with self.assertRaises(AssertionError):
                candidate.resource_gate(protocol, 0.)
        with patch.object(candidate.base, 'memory_free_percent', return_value=70.), \
             patch.object(candidate.base, 'peak_rss', return_value=1100), \
             patch.object(candidate.time, 'monotonic', return_value=20.):
            with self.assertRaises(AssertionError):
                candidate.resource_gate(protocol, 0.)

    def test_guard_order_spans_replays_label_opening_and_completion(self):
        source = inspect.getsource(candidate.main)
        tree = ast.parse(source)
        statements = tree.body[0].body
        guarded = next(node for node in statements if isinstance(node, ast.Try)).body
        rendered = [ast.unparse(node) for node in guarded]
        before = next(i for i, node in enumerate(rendered) if "report['resources_before_prediction_seal']" in node)
        seal = next(i for i, node in enumerate(rendered) if "(out / 'prediction_seal.json').write_text" in node)
        labels = next(i for i, node in enumerate(rendered) if node.startswith('truth = read_screen_labels'))
        after = next(i for i, node in enumerate(rendered) if "report['resources_after_scoring']" in node)
        complete = next(i for i, node in enumerate(rendered) if "report.update(status='complete'" in node)
        self.assertLess(before, seal)
        self.assertLess(seal, labels)
        self.assertLess(labels, after)
        self.assertLess(after, complete)
        self.assertEqual(rendered[before].split(' = ')[1], 'resource_gate(protocol, started)')
        self.assertEqual(rendered[after].split(' = ')[1], 'resource_gate(protocol, started)')

    def test_pure_mathematics_and_target_scope_functions_unchanged(self):
        from scripts import screen_canonical_pooled as original
        for name in ['early_splits', 'read_screen_labels', 'scores', 'select_candidate', 'predict_replay']:
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(candidate, name)))),
                             ast.dump(ast.parse(inspect.getsource(getattr(original, name)))))


if __name__ == '__main__':
    unittest.main()
