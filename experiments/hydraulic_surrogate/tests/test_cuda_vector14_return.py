"""Synthetic-only native vector-tree arithmetic and archive integrity checks."""
import copy
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from scripts.verify_cuda_vector14_smoke_return import json_forest_prediction,safe_import


def model(tree:dict)->dict:
    """Build a tiny native-shaped fixture without training a model."""
    return {'learner':{'learner_model_param':{'num_target':'14','num_feature':'664','base_score':'[0.5]'},
            'objective':{'name':'binary:logistic'},'gradient_booster':{'model':{
                'gbtree_model_param':{'num_trees':'8'},'trees':[copy.deepcopy(tree)for _ in range(8)]}}}}


class Vector14ReturnTests(unittest.TestCase):
    def test_fourteen_leaf_values_and_float32_accumulation(self):
        leaf=np.arange(14,dtype=np.float32)/100
        tree={'tree_param':{'num_nodes':'1','size_leaf_vector':'14'},'left_children':[-1],
              'right_children':[0],'split_indices':[0],'split_conditions':[0], 'leaf_weights':leaf.tolist()}
        actual=json_forest_prediction(model(tree),np.zeros((2,664),dtype=np.float32))
        margin=np.zeros((2,14),dtype=np.float32)
        for _ in range(8):margin+=leaf
        expected=1/(1+np.exp(-margin))
        self.assertEqual(actual.dtype,np.float32)
        self.assertEqual(actual.tobytes(),expected.tobytes())

    def test_split_boundary_and_leaf_index_mapping(self):
        tree={'tree_param':{'num_nodes':'3','size_leaf_vector':'14'},'left_children':[1,-1,-1],
              'right_children':[2,1,0],'split_indices':[4,0,0],'split_conditions':[.5,0,0],
              'leaf_weights':[.1]*14+[-.1]*14}
        x=np.zeros((2,664),dtype=np.float32);x[:,4]=[.49,.5]
        actual=json_forest_prediction(model(tree),x)
        self.assertTrue(np.all(actual[0]<.5));self.assertTrue(np.all(actual[1]>.5))

    def test_scalar_leaves_and_wrong_tree_count_rejected(self):
        tree={'tree_param':{'num_nodes':'1','size_leaf_vector':'1'},'left_children':[-1],
              'right_children':[0],'split_indices':[0],'split_conditions':[0], 'leaf_weights':[0]*14}
        with self.assertRaisesRegex(ValueError,'vector size'):
            json_forest_prediction(model(tree),np.zeros((1,664),dtype=np.float32))
        saved=model(tree);saved['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees']='112'
        with self.assertRaisesRegex(ValueError,'8 shared'):
            json_forest_prediction(saved,np.zeros((1,664),dtype=np.float32))

    def test_foreign_archive_namespace_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);archive=root/'wrong.zip'
            with zipfile.ZipFile(archive,'w')as z:z.writestr('../escaped','bad')
            with self.assertRaisesRegex(ValueError,'allowlist'):safe_import(archive,root/'imported')
            self.assertFalse((root/'imported').exists())


if __name__=='__main__':unittest.main()
