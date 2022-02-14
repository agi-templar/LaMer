#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmoputh.edu) &
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

import re
import string
from typing import Any, Generator, List, Union

import numpy as np
import torch
from tqdm import tqdm


def batch_it(seq: List[str], num: int = 1) -> Generator:
    assert num > 0
    batch: List[str] = []
    for e in tqdm(seq):
        if len(batch) == num:
            yield batch
            batch = [e]
        else:
            batch.append(e)
    yield batch


def get_sents_from_file(file_path: str) -> List[str]:
    """Load sentences from the file

    :param file_path:
    :return:
    """
    lines = open(file_path, 'r', encoding='utf-8').readlines()[:]
    features = []
    for _, line in enumerate(lines):
        seq = line.strip('\n')
        features.append(seq)
    return features


def normalize(s: str) -> str:
    """
    String normalizer.
    """
    if "_num_" in s.split(' '):
        return ''

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    def nt(text: str) -> str:
        return re.sub(r" n't", "n't", text)

    return white_space_fix(remove_punc(lower(nt(s))))


def cosine_similarity(
    a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """Compute the cosine similarity between two tensors.

    :param a:
    :param b:
    :return:
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def iterate_batches_lm(
    sents: List[str], sents_embedding: np.ndarray, batch_size: int
) -> Generator:
    """
    support iterate the sents (and its embeddings) by batches
    :param sents:
    :param sents_embedding:
    :param batch_size:
    :return:
    """
    assert isinstance(sents, list)
    assert isinstance(batch_size, int)

    ofs = 0
    while True:
        batch = (
            sents[ofs * batch_size:(ofs + 1) * batch_size],
            sents_embedding[ofs * batch_size:(ofs + 1) * batch_size]
        )
        if len(batch[0]) <= 1 or len(batch[1]) <= 1:
            break
        yield batch
        ofs += 1


def iterate_batches_lm_kg(
    sents: List[str], sents_embedding: np.ndarray, sents_ent: Any, batch_size: int
) -> Generator:
    """
    support iterate the sents (and its embeddings and kg) by batches
    :param sents:
    :param sents_embedding:
    :param sents_ent:
    :param batch_size:
    :return:
    """
    assert isinstance(sents, list)
    assert isinstance(batch_size, int)

    ofs = 0
    while True:
        batch = (
            sents[ofs * batch_size:(ofs + 1) * batch_size],
            sents_embedding[ofs * batch_size:(ofs + 1) * batch_size],
            sents_ent[ofs * batch_size:(ofs + 1) * batch_size]
        )
        if len(batch[0]) <= 1 or len(batch[1]) <= 1 or len(batch[2]) <= 1:
            break
        yield batch
        ofs += 1
