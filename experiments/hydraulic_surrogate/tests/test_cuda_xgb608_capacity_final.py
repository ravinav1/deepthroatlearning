"""Check exact final sampling, nested blend and fresh-smoke provenance."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scripts.train_cuda_xgb608_capacity_final import final_sample,verify_fresh_smoke,digest
from scripts.export_cuda_xgb608_capacity_final import compose
from scripts.launch_cuda_xgb608_capacity_final_dataset import find_exact
from sintef_project.postprocess import project_vinje


class CapacityFinalTests(unittest.TestCase):
    def test_exact_original_final_sample(self):
        source=Path('artifacts/lgb-normalized-lp-context-joint-final-001/train_sample_final.npy')
        actual=final_sample(2922*168,digest(source),np)
        np.testing.assert_array_equal(actual,np.load(source,allow_pickle=False))
        self.assertEqual(len(actual),196358)
        with self.assertRaises(ValueError):final_sample(2922*168,'0'*64,np)

    def test_nested_quarter_blend_preserves_projected_anchor(self):
        units=['Vinje_G1','Vinje_G2','Vinje_G3']
        anchor=project_vinje(np.tile(np.asarray([[.3,.8,.2],[.6,.4,.9]]),(84,1)),units)
        raw=np.tile(np.asarray([[.9,.2,.8],[.3,.1,.2]],dtype=np.float32),(84,1))
        np.testing.assert_array_equal(compose(anchor,raw,units),project_vinje(.75*anchor+.25*raw.astype(np.float64),units))
        with self.assertRaises(ValueError):compose(anchor,np.full_like(raw,1.1),units)

    def test_real_fresh_smoke_proof_and_tampering(self):
        source=Path('artifacts/verify-cuda-xgb608-capacity-2022-return-001/download/cuda-xgb608-capacity-smoke-002/manifest.json')
        validation_path=Path('docs/cuda_xgb608_capacity_protocol.json');validation=json.loads(validation_path.read_text())
        smoke=json.loads(source.read_text())
        protocol={'validation_protocol_sha256':digest(validation_path),'reference_cuda':smoke['cuda']}
        self.assertEqual(verify_fresh_smoke(source,digest(source),validation,protocol,Path('scripts')),smoke)
        bad=deepcopy(protocol);bad['reference_cuda']['requested_device']='cuda:1'
        with self.assertRaises(ValueError):verify_fresh_smoke(source,digest(source),validation,bad,Path('scripts'))
        with tempfile.TemporaryDirectory()as folder:
            copied=Path(folder)/'smoke';shutil.copytree(source.parent,copied)
            with (copied/'synthetic_smoke.ubj').open('ab')as stream:stream.write(b'tampered')
            with self.assertRaises(ValueError):verify_fresh_smoke(copied/'manifest.json',digest(source),validation,protocol,Path('scripts'))

    def test_duplicate_code_discovery_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);(root/'a').mkdir();(root/'b').mkdir();first=root/'a'/'protocol.json';first.write_text('{}')
            self.assertEqual(find_exact(root,'protocol.json',digest(first)),first.resolve())
            (root/'b'/'protocol.json').write_text('{}')
            with self.assertRaises(ValueError):find_exact(root,'protocol.json',digest(first))


if __name__=='__main__':unittest.main()
