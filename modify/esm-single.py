# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import string

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np
import esm
import math

from IPython import embed
from multiprocessing import Pool

torch.set_num_threads(1)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )

    # fmt: off
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=0,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    # fmt: on
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def mmp(data, muts):

    token_probs, sequence, alphabet, positions, offset_idx = data
    score = torch.zeros(1)

    assert len(muts) == len(positions)
    for i,mut in enumerate(muts):
        wt, idx, mt = mut[0], int(mut[1:-1])-1+offset_idx, mut[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
        score += token_probs[1+idx, mt_encoded] - token_probs[1+idx, wt_encoded]

    return score.item()


def main(args):

    df = pd.read_csv(args.dms_output, index_col=0)
    df.mutant = df.mutant.apply(lambda x: x.split(';') if x!='wt' else 'wt')

    # inference for each model
    for model_location in args.model_location:

        if model_location[3] == '1' or model_location[3]=='_':
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
        elif model_location[3] == '2':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        model.eval()
        model = model.cuda()

        batch_converter = alphabet.get_batch_converter()

        data = [
            ("protein1", args.sequence),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        scores = []

        for muts in df.mutant.values:
            
            if muts == 'wt':
                scores.append(0.0)
            
            else:
                positions = [int(mut[1:-1]) for mut in muts]

                batch_tokens_masked = batch_tokens.clone()
                for pos in positions:
                    idx = pos-1+args.offset_idx
                    batch_tokens_masked[0, 1+idx] = alphabet.mask_idx
                

                batch_tokens_masked = batch_tokens_masked.cuda()

                with torch.no_grad():
                    token_probs = torch.log_softmax(
                            model(batch_tokens_masked)["logits"], dim=-1
                        )
                    token_probs = token_probs[0].cpu().detach()

                data = token_probs, args.sequence, alphabet, positions, args.offset_idx
                score = mmp(data, muts)

                scores.append(score)   
        
        df[model_location] = scores


    df.mutant = df.mutant.apply(lambda x: ';'.join(x))
    df.to_csv(args.dms_output)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)