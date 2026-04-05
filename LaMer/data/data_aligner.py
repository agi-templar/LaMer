#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmoputh.edu)
#         Chongyang Gao (chongyang.gao.gr@dartmouth.edu)
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

import os
import pickle
import random
from abc import ABC
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, List, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from LaMer.data.scene_graph import batch_extract_entities, compute_sas
from LaMer.data.utils import get_sents_from_file, iterate_batches_lm, normalize

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataAligner(ABC):

    def __init__(
        self,
        config: Any,
        align_method: str,
        use_cache: bool = True,
        num_samples: int = -1
    ):
        super(DataAligner, self).__init__()
        assert align_method in ['random', 'lm', 'lm_kg']

        self.config = config
        self.align_method = align_method

        # Support different config field names across tasks
        src_field = next(
            f for f in ['pos_file_name', 'formal_file_name', 'left_file_name']
            if hasattr(self.config, f)
        )
        tgt_field = next(
            f for f in ['neg_file_name', 'informal_file_name', 'right_file_name']
            if hasattr(self.config, f)
        )
        self.src_data = self._preprocess(getattr(self.config, src_field))
        self.tgt_data = self._preprocess(getattr(self.config, tgt_field))

        if num_samples != -1:
            num_samples = min(num_samples, min(len(self.src_data), len(self.tgt_data)))
            self.src_data = self.src_data[:num_samples]
            self.tgt_data = self.tgt_data[:num_samples]

        if align_method == 'lm' or align_method == 'lm_kg':
            # the LM we use to mine paraphrases
            self.lm_model = SentenceTransformer('all-MiniLM-L6-v2')

            src_cache_name, tgt_cache_name = 'cached_emb_src_sents.pickle',\
                                             'cached_emb_tgt_sents.pickle'
            if use_cache:
                if os.path.exists(
                    os.path.join(self.config.cache_path, src_cache_name)
                ):
                    with open(
                        os.path.join(self.config.cache_path, src_cache_name), 'rb'
                    ) as f:
                        self.src_emb = pickle.load(f)
                else:
                    print(
                        "You choose use_cache but cache does not exist. "
                        "Now build for src_sents ..."
                    )
                    self.src_emb = self._build_emb(self.src_data, src_cache_name)

                if os.path.exists(
                    os.path.join(self.config.cache_path, tgt_cache_name)
                ):
                    with open(
                        os.path.join(self.config.cache_path, tgt_cache_name), 'rb'
                    ) as f:
                        self.tgt_emb = pickle.load(f)
                else:
                    print(
                        "You choose use_cache but cache does not exist. "
                        "Now build for tgt_sents ..."
                    )
                    self.tgt_emb = self._build_emb(self.tgt_data, tgt_cache_name)
                print("Loaded")

            else:
                print("You choose not use_cache, so we update cache for src_sents ...")
                self.src_emb = self._build_emb(self.src_data, src_cache_name)
                print("You choose not use_cache, so we update cache for tgt_sents ...")
                self.tgt_emb = self._build_emb(self.tgt_data, tgt_cache_name)
                print(
                    "Cache building finished! "
                    "In the future you can set use_cache to True."
                )

    def _build_emb(self, sents: List[str], cache_name: str) -> torch.Tensor:
        """Build embeddings for sentences with a LM.

        :param sents: a list of input sentences
        :param cache_name: the name of the cache file
        :return:
        """
        embeddings = self.lm_model.encode(sents)

        print("Saving to pickle ...")
        if not os.path.exists(self.config.cache_path):
            os.makedirs(self.config.cache_path)

        with open(os.path.join(self.config.cache_path, cache_name), 'wb') as f:
            pickle.dump(embeddings, f, protocol=4)
            print("Saved cache!")

        return embeddings

    def get_src_data(self) -> List[str]:
        return self.src_data

    def get_tgt_data(self) -> List[str]:
        return self.tgt_data

    def get_src_emb(self) -> Union[np.ndarray, torch.Tensor, None]:
        if self.align_method in ['lm', 'lm_kg']:
            return self.src_emb
        else:
            return None

    def get_tgt_emb(self) -> Union[np.ndarray, torch.Tensor, None]:
        if self.align_method in ['lm', 'lm_kg']:
            return self.tgt_emb
        else:
            return None

    @staticmethod
    def _preprocess(file_path: str) -> List:
        """Get a list of sentences from the specified file_path.

        :param file_path: path to the styled sentences
        :return: a list of sentences (src or tgt) after preprocessing
        """
        sents_list = get_sents_from_file(file_path)

        print("The length of sents (before preprocessing)", len(sents_list))

        # normalize and filter the data
        sents_list = [normalize(sent) for sent in sents_list]
        sents_list = [sent for sent in sents_list if len(sent.split(' ')) > 1]
        sents_list = list(set(sents_list))

        print("The length of sents (after preprocessing)", len(sents_list))

        return sents_list

    @staticmethod
    def _get_random(src_sents: str, tgt_sents: List[str],
                    n: int) -> List[dict]:
        """Single process version of getting a random mapped tgt sentences for src sentences

        :param src_sents:
        :param tgt_sents:
        :param n:
        :return:
        """
        rows = []
        for tgt_idx in [random.randint(0, len(tgt_sents) - 1) for _ in range(n)]:
            rows.append({'source': src_sents, 'target': tgt_sents[tgt_idx]})
        return rows

    @staticmethod
    def _compute_lm_toppk(
        src_sents: List[str], src_emb: Union[np.ndarray, torch.Tensor],
        tgt_sents: List[str], tgt_emb: Union[np.ndarray,
                                             torch.Tensor], topp: float, topk: int
    ) -> List[dict]:
        """Single process version of picking by top-p and top-k for LM alignment.

        :param src_sents:
        :param src_emb:
        :param tgt_sents:
        :param tgt_emb:
        :param topp: float, the top-p value
        :param topk: int, the top-k value
        :return:
        """
        _temp_dicts = []

        cosine_matrix = util.pytorch_cos_sim(src_emb, tgt_emb).cpu().numpy()

        # filter out zeros, to speed up
        cosine_matrix = np.around(cosine_matrix, decimals=3)
        tgt_sents = np.array(tgt_sents, dtype=str)

        top_k_indices = [
            np.argpartition(-row, topk)[:topk] for row in cosine_matrix
        ]  # fast nlargest
        top_k_scores = [-np.partition(-row, topk)[:topk] for row in cosine_matrix]

        for src_sent_idx, src_sent in enumerate(src_sents):
            cur_topk_idx, cur_topk_sc = top_k_indices[src_sent_idx], top_k_scores[
                src_sent_idx]
            picked_sents = tgt_sents[cur_topk_idx].tolist()
            masked_ids = [idx for idx, sc in enumerate(cur_topk_sc) if sc > topp]
            picked_sents = [
                sent for idx, sent in enumerate(picked_sents) if idx in masked_ids
            ]
            picked_scores = [
                sc for idx, sc in enumerate(cur_topk_sc) if idx in masked_ids
            ]
            if picked_sents:
                _temp_dict = {
                    'source': src_sent,
                    'target': picked_sents,
                    'similarity_score': picked_scores
                }
                _temp_dicts.append(_temp_dict)

        return _temp_dicts

    def align_by_random(
        self, topk: int, output_root_dir: str, output_file_name: str
    ) -> pd.DataFrame:
        """Align the source and target style sentences by random mapping.

        :param topk: topk value
        :param output_root_dir: the folder that stores the output csv
        :param output_file_name: the file name of the output csv
        :return: a pandas.DataFrame with aligned sentences
        """
        assert len(self.src_data) != 0 and len(
            self.tgt_data
        ) != 0, "The src/tgt sentences are None!"

        if not os.path.exists(self.config.cache_path):
            os.makedirs(self.config.cache_path)

        _part_get_random = partial(self._get_random, n=topk)
        args = [(src_sent, self.tgt_data) for src_sent in self.src_data]

        with Pool(cpu_count()) as proc:
            _temp_rows_list = list(
                tqdm(proc.starmap(
                    _part_get_random,
                    args,
                ), total=len(self.src_data))
            )
            all_rows = [row for rows in _temp_rows_list for row in rows]
            out_df = pd.DataFrame(all_rows, columns=['source', 'target'])

        if not os.path.exists(output_root_dir):
            os.makedirs(output_root_dir)

        out_df.to_csv(os.path.join(output_root_dir, output_file_name), index=False)
        return out_df

    def align_by_LM(
        self, topp: float, topk: int, output_root_dir: str, output_file_name: str
    ) -> pd.DataFrame:
        """Align the source and target style sentences by LM embeddings.

        :param topp: topp value
        :param topk: topk value
        :param output_root_dir: the folder that stores the output csv
        :param output_file_name: the file name of the output csv
        :return: a pandas.DataFrame with aligned sentences
        """
        assert len(self.src_data) != 0 and len(
            self.tgt_data
        ) != 0, "The src/tgt sentences are None!"

        all_dicts = []
        batch_size = min(
            self.config.lm_batch_size, min(len(self.src_data), len(self.tgt_data))
        )

        _part_compute_emb_toppk = partial(
            self._compute_lm_toppk,
            tgt_emb=self.tgt_emb,
            topp=topp,
            topk=topk
        )

        print("Now working on top p and top k picking ...")
        for batched_src_sents in tqdm(
            iterate_batches_lm(self.src_data, self.src_emb, batch_size),
            total=len(self.src_data) // batch_size
        ):
            src_sents, src_embs = batched_src_sents
            _temp_dicts = _part_compute_emb_toppk(src_sents, src_embs, self.tgt_data)
            all_dicts.extend(_temp_dicts)

        out_df = pd.DataFrame(all_dicts)

        if not os.path.exists(output_root_dir):
            os.makedirs(output_root_dir)

        out_df.to_csv(os.path.join(output_root_dir, output_file_name), index=False)
        return out_df

    def align_by_LM_KG(
        self, output_root_dir: str, output_file_name: str
    ) -> pd.DataFrame:
        """Align sentences by LM embeddings + Scene Alignment Score (SAS).

        Steps (per paper Section 2.2):
        1. Use LM embeddings to pick top-k candidate targets per source.
        2. Extract scene entities from source and each candidate target.
        3. Compute SAS for each pair; keep those above threshold p.

        :param output_root_dir: the folder that stores the output csv
        :param output_file_name: the file name of the output csv
        :return: a pandas.DataFrame with aligned sentences
        """
        assert self.align_method == 'lm_kg', \
            "align_by_LM_KG requires align_method='lm_kg'"

        topk = self.config.lm_kg_topk
        topp = self.config.lm_kg_topp
        batch_size = self.config.lm_kg_batch_size
        beta = getattr(self.config, 'beta', 0.01)

        print("Step 1: LM-based top-k candidate selection ...")
        lm_candidates = []
        _part_compute = partial(
            self._compute_lm_toppk,
            tgt_emb=self.tgt_emb,
            topp=0.0,  # no p-filtering at this stage
            topk=topk
        )
        for batched in tqdm(
            iterate_batches_lm(self.src_data, self.src_emb, batch_size),
            total=len(self.src_data) // batch_size
        ):
            src_sents, src_embs = batched
            _temp = _part_compute(src_sents, src_embs, self.tgt_data)
            lm_candidates.extend(_temp)

        print("Step 2: Scene graph extraction and SAS filtering ...")
        all_rows = []
        for item in tqdm(lm_candidates):
            src_sent = item['source']
            tgt_sents = item['target']
            sim_scores = item['similarity_score']

            src_entities = batch_extract_entities([src_sent])[0]
            if not src_entities:
                continue

            for tgt_sent, sim_sc in zip(tgt_sents, sim_scores):
                tgt_entities = batch_extract_entities([tgt_sent])[0]
                tgt_len = len(tgt_sent.split())
                sas = compute_sas(src_entities, tgt_entities, tgt_len, beta)

                if sas >= topp:
                    all_rows.append({
                        'source': src_sent,
                        'target': tgt_sent,
                        'similarity_score': sim_sc,
                        'sas_score': sas
                    })

        out_df = pd.DataFrame(all_rows)

        if not os.path.exists(output_root_dir):
            os.makedirs(output_root_dir)

        out_df.to_csv(os.path.join(output_root_dir, output_file_name), index=False)
        print(f"LM+KG alignment done. {len(out_df)} pairs saved.")
        return out_df
