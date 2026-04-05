#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmouth.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Scene graph extraction and Scene Alignment Score (SAS) computation.

Uses SceneGraphParser (https://github.com/vacancy/SceneGraphParser) to extract
subject-relation-object triplets. Only end nodes (subj & obj) are kept as
scene entities. SAS is an F-beta score normalized by target sentence length.

Reference: Section 2.2 of the paper (Equation 1).
"""

import warnings
from typing import List

try:
    import sng_parser
    HAS_SNG_PARSER = True
except ImportError:
    HAS_SNG_PARSER = False
    warnings.warn(
        "sng_parser not installed. Install via: "
        "pip install SceneGraphParser. "
        "LM+KG alignment will not be available.",
        stacklevel=2
    )


def extract_scene_entities(sentence: str) -> List[str]:
    """Extract scene entities from a sentence using SceneGraphParser.

    Scene entities are the subject and object nodes from
    subject-relation-object triplets parsed from the sentence.

    :param sentence: input sentence
    :return: list of entity strings (lowercased, may contain duplicates)
    """
    if not HAS_SNG_PARSER:
        raise RuntimeError("sng_parser is required for scene graph extraction.")

    graph = sng_parser.parse(sentence)
    entities = []
    for rel in graph['relations']:
        subj_idx = rel['subject']
        obj_idx = rel['object']
        entities.append(graph['entities'][subj_idx]['head'].lower())
        entities.append(graph['entities'][obj_idx]['head'].lower())

    # If no relations found, fall back to all entity heads
    if not entities:
        for ent in graph['entities']:
            entities.append(ent['head'].lower())

    return entities


def compute_sas(
    src_entities: List[str],
    tgt_entities: List[str],
    tgt_sent_len: int,
    beta: float = 0.01
) -> float:
    """Compute Scene Alignment Score (SAS) between source and target entities.

    SAS(s^src || s^tgt) = (1 / |s^tgt|) * ((1 + beta^2) * precision * recall)
                          / (beta^2 * precision + recall)

    where precision = |overlap| / |tgt_entities|
          recall    = |overlap| / |src_entities|

    :param src_entities: scene entities from source sentence
    :param tgt_entities: scene entities from target sentence
    :param tgt_sent_len: number of words in the target sentence
    :param beta: weighting factor (recall is beta times as important as precision)
    :return: SAS score (float)
    """
    if not src_entities or not tgt_entities or tgt_sent_len == 0:
        return 0.0

    src_set = set(src_entities)
    tgt_set = set(tgt_entities)
    overlap = src_set & tgt_set

    if not overlap:
        return 0.0

    precision = len(overlap) / len(tgt_set)
    recall = len(overlap) / len(src_set)

    if precision + recall == 0:
        return 0.0

    beta_sq = beta**2
    f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    sas = f_beta / tgt_sent_len

    return sas


def batch_extract_entities(sentences: List[str]) -> List[List[str]]:
    """Extract scene entities for a batch of sentences.

    :param sentences: list of input sentences
    :return: list of entity lists, one per sentence
    """
    return [extract_scene_entities(s) for s in sentences]
