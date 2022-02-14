#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmoputh.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from LaMer.data import DataAligner

if __name__ == '__main__':
    from config import TestConfig
else:  # pytest
    from test.config import TestConfig


def test_yelp_random_align(num_sample=100, topk=2):
    config = TestConfig['yelp_pos2neg']
    aligner = DataAligner(config, 'random', use_cache=True, num_samples=num_sample)
    assert len(aligner.get_src_data()) != 0
    assert len(aligner.get_tgt_data()) != 0
    assert aligner.get_src_emb() is None
    assert aligner.get_tgt_emb() is None
    df = aligner.align_by_random(topk, 'assets/yelp/yelp_p2n_rd_test', 'top2.csv')
    assert list(df.columns) == ['source', 'target']


def test_yelp_lm_align(num_sample=100, topp=0.3, topk=3):
    config = TestConfig['yelp_pos2neg']
    aligner = DataAligner(config, 'lm', use_cache=True, num_samples=num_sample)
    assert len(aligner.get_src_data()) != 0
    assert len(aligner.get_tgt_data()) != 0
    assert aligner.get_src_emb() is not None
    assert aligner.get_tgt_emb() is not None
    df = aligner.align_by_LM(
        topp, topk, 'assets/yelp/yelp_p2n_lm_test', 'top2_top003.csv'
    )
    assert list(df.columns) == ['source', 'target', 'similarity_score']


def test_yelp_lm_kg_align(size=10000, num=8, sleep=0.1):
    # aligner = DataAligner('yelp_pos2neg', 'random', use_cache=True)
    # aligner.align_by_LM_KG('yelp_p2n_lm', 'top2_top003.csv')
    pass


if __name__ == '__main__':
    # test_config = TestConfig['yelp_pos2neg']
    test_yelp_random_align()
    test_yelp_lm_align()

    test_yelp_random_align()
    test_yelp_lm_align()
