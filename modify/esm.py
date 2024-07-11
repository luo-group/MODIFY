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
        "--dms-input",
        type=pathlib.Path,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
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
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    parser.add_argument(
        "--protein",
        type=str,
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA in a3m format (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to select from the start of the MSA"
    )
    # fmt: on
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def mut2mask(muts):
    mask = []
    for mut in muts:
        wt, idx, mt = mut[0], int(mut[2:-1]), mut[-1]
        mask.append(str(int(wt!=mt)))
    return int(''.join(mask), 2)


def init_worker(data):
    global token_probs_dict, sequence, alphabet, positions
    token_probs_dict, sequence, alphabet, positions = data


def mmp_multi(data):

    muts, mask = data
    score = torch.zeros(1)

    token_probs = token_probs_dict[mask]

    assert len(muts) == token_probs.shape[0]
    for i,mut in enumerate(muts):
        wt, idx, mt = mut[0], int(mut[2:-1])-1, mut[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
        score += token_probs[i, mt_encoded] - token_probs[i, wt_encoded]

    return score.item()


def main(args):

    if args.protein == 'GB1':
        positions = [39, 40, 41, 54]
    positions = [pos-1 for pos in positions]

    df = pd.read_csv(args.dms_output, index_col=0)
    # df.mutant = df.mutant.apply(lambda x: x.split(','))
    df.mutant = df.mutant.apply(lambda x: x.split(';'))

    df['masks'] = df.mutant.apply(lambda x: mut2mask(x))

    masks_option = list(df.masks.value_counts().index)

    workload = df[['mutant', 'masks']].values.tolist()


    tot = len(df)
    batch_size = 1
    num_batches = int(tot / batch_size)

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

        if args.scoring_strategy == "masked-marginals":
            scores = []
            
            token_probs_dict = dict({})

            input_tokens = batch_tokens.repeat(len(masks_option), 1)
            for it,i in enumerate(masks_option):

                mask = [int(a) for a in "{0:b}".format(i)]

                batch_tokens_masked = batch_tokens.clone()
                for m,pos in zip(mask,positions):
                    if m==1:
                        batch_tokens_masked[0, 1+pos] = alphabet.mask_idx
                
                input_tokens[it, :] = batch_tokens_masked[0, :]

            batch_size = 64
            num_batches = math.ceil(len(masks_option)/batch_size)

            input_tokens = input_tokens.cuda()
            results = torch.zeros((len(masks_option), len(args.sequence)+2, 33))
            with torch.no_grad():
                for b in tqdm(range(num_batches)):
                    st, ed = b*batch_size, min((b+1)*batch_size, len(masks_option))
                    token_probs = torch.log_softmax(
                        model(input_tokens[st:ed])["logits"], dim=-1
                    )
                    results[st:ed] = token_probs[:].cpu().detach()
            for i in range(len(masks_option)):
                token_probs_dict[masks_option[i]] = results[i,[1+pos for pos in positions],:].cpu().detach()

            data = token_probs_dict, args.sequence, alphabet, positions

            with Pool(10, initializer=init_worker, initargs=(data,)) as pool:
                scores = list(tqdm(pool.map(mmp_multi,workload), total=len(df)))     
            
            df[model_location] = scores


    df.mutant = df.mutant.apply(lambda x: ';'.join(x))
    df.to_csv(args.dms_output)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)