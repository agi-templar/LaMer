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

from types import SimpleNamespace

TestConfig = {
    'yelp_pos2neg':
    SimpleNamespace(
        **{
            'pos_file_name': "assets/yelp/raw/train.pos",
            'neg_file_name': "assets/yelp/raw/train.neg",
            'cache_path': './assets/yelp/raw/p2n_cached_test/',
            'random_topk': 2,
            'random_cache_file': 'yelp_p2n_random.csv',
            'random_root_name': 'yelp_random',

            # 'random_topk': 10,
            # 'random_cache_file': 'yelp_p2n_random_plot.csv',
            # 'random_root_name': 'yelp_random_plot',
            'lm_topk': 2,
            'lm_topp': 0.03,
            'lm_batch_size': 10000,

            # 'lm_topk': 10,
            # 'lm_topp': 0.03,
            # 'lm_batch_size': 10000,
            # 'lm_cache_file': 'yelp_p2n_lm_top10_top003_plot.csv',
            # 'lm_root_name': 'yelp_lm_top10_top003_plot',
            'beta': 0.01,
            'lm_kg_topk': 50,
            'lm_kg_topp': 0.6,
            'lm_kg_batch_size': 1000,

            # 'beta': 0.01,
            # 'lm_kg_topk': 2800,
            # 'lm_kg_topp': 0.5,
            # 'lm_kg_batch_size': 1000,
            # 'lm_kg_cache_file': "yelp_p2n_lm_kg_tok2800_top05_beta001_plot.csv",
            # 'lm_kg_root_name': 'yelp_lm_kg_tok2800_top05_beta001_plot',
        }
    ),
    'yelp_neg2pos':
    SimpleNamespace(
        **{
            'pos_file_name': "assets/yelp/raw/train.pos",
            'neg_file_name': "assets/yelp/raw/train.neg",
            'cache_path': './assets/yelp/raw/p2n_cached/',
            't2t_cache_path': "./assets/yelp/raw/t2t/pos2neg/",
            'rl_cache_path': "./assets/yelp/raw/rl/pos2neg/",
            'random_topk': 3,
            'random_cache_file': 'yelp_n2p_random.csv',
            'random_root_name': 'yelp_random',
            'lm_topk': 3,
            'lm_topp': 0.03,
            'lm_batch_size': 10000,
            'lm_cache_file': 'yelp_n2p_lm_top3_top003.csv',
            'lm_root_name': 'yelp_lm_top3_top003',
            'lm_kg_topk': 100,
            'lm_kg_topp': 0.3,
            'lm_kg_batch_size': 4000,
            'lm_kg_cache_file': "yelp_n2p_lm_kg_tok100_top03.csv",
            'lm_kg_root_name': 'yelp_lm_kg_tok100_top03',
        }
    ),
    'formal_music_f2i':
    SimpleNamespace(
        **{
            'formal_file_name': "data/GYAFC_Corpus/Entertainment_Music/train/formal",
            'informal_file_name':
            "data/GYAFC_Corpus/Entertainment_Music/train/informal",
            'cache_path': './data/formal_music/f2i_cached/',
            't2t_cache_path': "./data/formal_music/t2t/f2i/",
            'rl_cache_path': "./data/formal_music/rl/f2i/",
            'random_topk': 3,
            'random_cache_file': 'formal_music_f2i_random.csv',
            'random_root_name': 'formal_music_random',
            'lm_topk': 3,
            'lm_topp': 0.03,
            'lm_batch_size': 10000,
            'lm_cache_file': 'formal_music_f2i_lm_top3_top003.csv',
            'lm_root_name': 'formal_music_f2i_top3_top003',
            'lm_kg_topk': 110,
            'lm_kg_topp': 0.015,
            'lm_kg_batch_size': 4000,
            'lm_kg_cache_file': "formal_music_f2i_lm_kg_tok100_top003.csv",
            'lm_kg_root_name': 'formal_music_f2i_lm_kg_tok100_top003',
        }
    ),
    'formal_music_i2f':
    SimpleNamespace(
        **{
            'formal_file_name': "data/GYAFC_Corpus/Entertainment_Music/train/informal",
            'informal_file_name': "data/GYAFC_Corpus/Entertainment_Music/train/formal",
            'cache_path': './data/formal_music/i2f_cached/',
            't2t_cache_path': "./data/formal_music/t2t/i2f/",
            'rl_cache_path': "./data/formal_music/rl/i2f/",
            'random_topk': 3,
            'random_cache_file': 'formal_music_i2f_random.csv',
            'random_root_name': 'formal_music_random',
            'lm_topk': 3,
            'lm_topp': 0.03,
            'lm_batch_size': 10000,
            'lm_cache_file': 'formal_music_i2f_lm_top3_top003.csv',
            'lm_root_name': 'formal_music_i2f_top3_top003',
            'lm_kg_topk': 110,
            'lm_kg_topp': 0.015,
            'lm_kg_batch_size': 4000,
            'lm_kg_cache_file': "formal_music_i2f_lm_kg_tok110_top0015.csv",
            'lm_kg_root_name': 'formal_music_i2f_lm_kg_tok110_top0015',
        }
    ),
    'formal_family_f2i':
    SimpleNamespace(
        **{
            'formal_file_name': "data/GYAFC_Corpus/Family_Relationships/train/formal",
            'informal_file_name':
            "data/GYAFC_Corpus/Family_Relationships/train/informal",
            'cache_path': './data/formal_family/f2i_cached/',
            't2t_cache_path': "./data/formal_family/t2t/f2i/",
            'rl_cache_path': "./data/formal_family/rl/f2i/",
            'random_topk': 2,
            'random_cache_file': 'formal_family_f2i_random.csv',
            'random_root_name': 'formal_family_random',
            'lm_topk': 2,
            'lm_topp': 0.03,
            'lm_batch_size': 10000,
            'lm_cache_file': 'formal_family_f2i_lm_top2_top003.csv',
            'lm_root_name': 'formal_family_f2i_top2_top003',
            'beta': 0.01,
            'lm_kg_topk': 500,
            'lm_kg_topp': 0.4,
            'lm_kg_batch_size': 1000,
            'lm_kg_cache_file': "formal_family_f2i_lm_kg_tok500_top04.csv",
            'lm_kg_root_name': 'formal_family_f2i_lm_kg_tok500_top04',
        }
    ),
    'formal_family_i2f':
    SimpleNamespace(
        **{
            'formal_file_name': "data/GYAFC_Corpus/Entertainment_Music/train/informal",
            'informal_file_name': "data/GYAFC_Corpus/Entertainment_Music/train/formal",
            'cache_path': './data/formal_music/i2f_cached/',
            't2t_cache_path': "./data/formal_music/t2t/i2f/",
            'rl_cache_path': "./data/formal_music/rl/i2f/",
            'random_topk': 3,
            'random_cache_file': 'formal_music_i2f_random.csv',
            'random_root_name': 'formal_music_random',
            'lm_topk': 3,
            'lm_topp': 0.03,
            'lm_batch_size': 10000,
            'lm_cache_file': 'formal_music_i2f_lm_top3_top003.csv',
            'lm_root_name': 'formal_music_i2f_top3_top003',
            'lm_kg_topk': 110,
            'lm_kg_topp': 0.015,
            'lm_kg_batch_size': 4000,
            'lm_kg_cache_file': "formal_music_i2f_lm_kg_tok110_top0015.csv",
            'lm_kg_root_name': 'formal_music_i2f_lm_kg_tok110_top0015',
        }
    ),
    'allsides_l2r':
    SimpleNamespace(
        **{
            'left_file_name': "data/allsides/left_output/",
            'right_file_name': "data/allsides/right_output/",
            'cache_path': './data/allsides/l2r_cached/',
            't2t_cache_path': "./data/allsides/t2t/l2r/",
            'rl_cache_path': "./data/allsides/rl/l2r/",
            'random_topk': 1,
            'random_cache_file': 'allsides_l2r_random.csv',
            'random_root_name': 'allsides_random',
            'lm_topk': 2,
            'lm_topp': 0.6,
            'lm_batch_size': 10000,
            'lm_cache_file': 'allsides_l2r_lm_top2_top06.csv',
            'lm_root_name': 'allsides_lm_top2_top06',
            'lm_kg_topk': 50,
            'lm_kg_topp': 0.4,
            'lm_kg_batch_size': 4000,
            'lm_kg_cache_file': "allsides_l2r_lm_kg_tok50_top04.csv",
            'lm_kg_root_name': 'allsides_lm_kg_tok50_top04',
        }
    ),
    'allsides_r2l':
    SimpleNamespace(
        **{
            'left_file_name': "data/allsides/right_output/",
            'right_file_name': "data/allsides/left_output/",
            'cache_path': './data/allsides/r2l_cached/',
            't2t_cache_path': "./data/allsides/t2t/r2l/",
            'rl_cache_path': "./data/allsides/rl/r2l/",
            'random_topk': 2,
            'random_cache_file': 'allsides_r2l_random.csv',
            'random_root_name': 'allsides_random',
            'lm_topk': 5,
            'lm_topp': 0.5,
            'lm_batch_size': 10000,
            'lm_cache_file': 'allsides_r2l_lm_top2_top05.csv',
            'lm_root_name': 'allsides_lm_top2_top05',
            'lm_kg_topk': 100000,
            'lm_kg_topp': 0.1,
            'lm_kg_batch_size': 4000,
            'lm_kg_cache_file': "allsides_r2l_lm_kg_tok20000_top01.csv",
            'lm_kg_root_name': 'allsides_lm_kg_tok20000_top01',
        }
    ),
}
