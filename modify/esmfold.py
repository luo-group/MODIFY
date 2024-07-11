import argparse
import os
import torch
import esm
import time
import pandas as pd
import math

from torch import inf

torch.set_num_threads(1)


def main(args):
    '''
    Revised from the script from ESM github repo: https://github.com/facebookresearch/esm#esmfold
    '''

    print(args.cuda)
    device = torch.device(f'cuda:{args.cuda}')

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    model = model.to(device)

    # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
    # Lower sizes will have lower memory requirements at the cost of increased speed.
    # model.set_chunk_size(128)

    sequence = args.sequence
    # Multimer prediction can be done with chains separated by ':'

    with torch.no_grad():

        output = model.infer_pdb(sequence)

        name = args.name
        with open(os.path.join(args.outputpath, f'{name}.pdb'), "w") as f:
            f.write(output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', '-c', default=0, type=int, help='cuda device')
    parser.add_argument('--sequence', '-s', type=str, required=True, help='sequence for prediction')
    parser.add_argument('--name', '-n', type=str, required=True, help='name for prediction')
    parser.add_argument('--outputpath', '-o', type=str, required=True, help='output file')
    args = parser.parse_args()

    main(args)