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

from multiprocessing import Pool

torch.set_num_threads(1)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
    
    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""

    for record in itertools.islice(SeqIO.parse(filename, "fasta"), 1):
        mapping = dict({})
        seq = record.seq
        cnt = 0
        for i,s in enumerate(seq):
            if s>='A' and s<='Z':
                mapping[i] = cnt
                cnt += 1
            else:
                mapping[i] = -1
    print(cnt)


    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa, mapping


def read_msa_nonfocus(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
    
    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""

    for record in itertools.islice(SeqIO.parse(filename, "fasta"), 1):
        mapping = dict({})
        seq = record.seq
        cnt = 0
        for i,s in enumerate(seq):
            mapping[i] = cnt
            cnt += 1
    print(cnt)


    msa = [
        (record.description, str(record.seq))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa, mapping


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


def label_row(row, sequence, token_probs, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # add 1 for BOS
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()


def mmp(muts, token_probs, sequence, alphabet):
    score = torch.zeros(1).cuda()
    for mut in muts:
        wt, idx, mt = mut[0], int(mut[2:-1]), mut[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
        score += token_probs[1 + idx, mt_encoded] - token_probs[1 + idx, wt_encoded]

    return score.item()


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


def mut2mask(muts):
    mask = []
    for mut in muts:
        wt, idx, mt = mut[0], int(mut[2:-1]), mut[-1]
        mask.append(str(int(wt!=mt)))
    return int(''.join(mask), 2)

def init_worker_msa(data):
    global token_probs_dict, sequence, alphabet, mapped_positions, original_sequence
    token_probs_dict, sequence, alphabet, mapped_positions, original_sequence = data

def mmp_msa_multi(data):

    muts, mask = data

    score = torch.zeros(1)

    # mask = []
    # for mut in muts:
    #     wt, idx, mt = mut[0], int(mut[2:-1]), mut[-1]
    #     mask.append(str(int(wt!=mt)))
    token_probs = token_probs_dict[mask]

    assert len(muts) == token_probs.shape[0]
    for i,mut in enumerate(muts):
        wt, _, mt = mut[0], int(mut[2:-1])-1, mut[-1]
        # idx = mapped_positions[i]
        # assert original_sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
        score += token_probs[i, mt_encoded] - token_probs[i, wt_encoded]

    return score.item()

def main(args):

    if args.protein == 'GB1':
        positions = [39, 40, 41, 54]
    positions = [pos-1 for pos in positions]

    df = pd.read_csv(args.dms_output, index_col=0)
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

        if isinstance(model, MSATransformer):
            
            msa_seed_list = range(5)
                
            for seed in msa_seed_list:
                print(seed)

                msa, mapping = read_msa(str(args.msa_path)+str(seed)+'.a2m', args.msa_samples)
                data = [msa]

                assert (
                    args.scoring_strategy == "masked-marginals"
                ), "MSA Transformer only supports masked marginal strategy"

                batch_labels, batch_strs, batch_tokens = batch_converter(data)

                token_probs_dict = dict({})
                input_tokens = batch_tokens.repeat(len(masks_option), 1, 1)
                for it,i in tqdm(enumerate(masks_option)):
                    mask = [int(a) for a in "{0:b}".format(i)]

                    # batch_tokens_masked = batch_tokens.clone()
                    for m,pos in zip(mask, positions):
                        if m == 1:
                            input_tokens[it, 0, 1+mapping[pos]] = alphabet.mask_idx  # mask out first sequence
                
                batch_size = 8
                num_batches = len(masks_option) // batch_size + 1
                print(len(masks_option), batch_size, num_batches)

                results = torch.zeros((len(masks_option), len(positions), 33))
                with torch.no_grad():
                    for b in tqdm(range(num_batches)):
                        st, ed = b*batch_size, min((b+1)*batch_size, len(masks_option))
                        token_probs = torch.log_softmax(
                            model(input_tokens[st:ed].cuda())["logits"], dim=-1
                        ).cpu().detach()
                        results[st:ed] = token_probs[:,0,[1+mapping[pos] for pos in positions],:]
                for i in range(len(masks_option)):
                    token_probs_dict[masks_option[i]] = results[i]
                
                data = token_probs_dict, data[0][0][1], alphabet, [mapping[pos] for pos in positions], args.sequence

                with Pool(10, initializer=init_worker_msa, initargs=(data,)) as pool:
                    scores = list(tqdm(pool.map(mmp_msa_multi, workload),total=len(df)))     
                    
                df[f'{model_location}_{seed}'] = scores

    df.mutant = df.mutant.apply(lambda x: ';'.join(x))
    df.to_csv(args.dms_output)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
