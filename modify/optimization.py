import os
import sys
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modify.utils import softmax_mask, get_alphabet, load_sequence, get_mask, init_worker_probability, calculate_probability

torch.set_num_threads(1)


class MODIFYOptimization:

    def __init__(self, protein='GB1', offset=1, positions=[39,40,41,54], masked_AAs=[], fitness_col='modify_fitness', seed=42, lr=0.1, B=1000, T=2000, path=''):
        
        self.protein = protein
        self.positions = positions
        self.num_position = len(positions)
        self.masked_AAs = masked_AAs
        self.offset = offset
        self.fitness_col = fitness_col
        self.seed = seed
        self.lr = lr
        self.B = B
        self.T = T

        # load alphabet
        self.alphabet, self.map_a2i, self.map_i2a = get_alphabet()

        # load wildtype sequence from fasta file
        self.wt_seq = load_sequence(f'{path}data/{self.protein}/wt.fasta')
        print(f'Sequence data for {self.protein} loaded...')
        print(f'starting sequence:', self.wt_seq)
        print(f'target residues:', ','.join([f'{self.wt_seq[pos-self.offset]}{pos}' for pos in self.positions]))

        # load mask
        self.mask = get_mask(self.positions, self.num_position, self.map_a2i, self.masked_AAs)
        print('Mask loaded...')
        print('Mask shape:', self.mask.shape)
        print('Mask:', self.mask)
        rows, cols = torch.where(self.mask == 0)
        if self.masked_AAs:
            print('Masked AAs:', ','.join([f'{self.positions[r]}{self.map_i2a[c]}' for r,c in zip(rows.tolist(), cols.tolist())]))
        else:
            print('No masked AAs')

        # load fitness data
        self.df = pd.read_csv(f'{path}data/{self.protein}/{self.protein}_zero.csv', index_col=0)
        self.fitnesses = self.df[self.fitness_col].values
        print('Fitness data loaded...')
        print(self.fitnesses)

    
    def parallel_optimization_default(self, lam_list, parallel=True, num_proc=60):

        if parallel:

            with Pool(num_proc) as pool:
                pool.map(self._optimization, lam_list, chunksize=1)
        else:
            for lam in tqdm(lam_list):
                self._optimization(lam)


    def _optimization(self, lam):
        """
        Optimization function under the default setting of MODIFY
        """

        if self.masked_AAs:
            writer = SummaryWriter(f'log/{self.protein}/runs_default/{self.protein}_{self.seed}_{lam}_{"".join(self.masked_AAs)}')
        else:
            writer = SummaryWriter(f'log/{self.protein}/runs_default/{self.protein}_{self.seed}_{lam}')

        lam_arr = (torch.ones(1)*lam).repeat(self.num_position) # The i-th element of lam_arr is equivalent to lam*alpha_i in the paper
        
        # Initialize phi
        np.random.seed(self.seed)
        random_array = np.random.rand(self.num_position, 20)
        phi = torch.tensor(random_array, dtype=torch.float32)

        eps = 1e-45

        for step in range(self.T):

            ''''Softmax with mask'''
            q = softmax_mask(phi, self.mask)

            '''sample'''
            samples = torch.vstack([torch.multinomial(q[i], self.B, replacement=True) for i in range(self.num_position)]) 
            indices = (samples * 20**(self.num_position-1-torch.arange(self.num_position).view(self.num_position,1).repeat(1,self.B))).sum(dim=0).tolist()
            
            dphi = torch.zeros((self.num_position,20))
            mean_fx = 0
            for i,x in enumerate(indices):
                
                fx = self.fitnesses[x]
                mean_fx += fx

                wx = fx

                delta = torch.zeros((self.num_position,20))
                delta[range(self.num_position),samples[:,i]] = 1

                dphi += wx * (delta - q)

            dphi /= self.B
            mean_fx /= self.B

            dh = torch.zeros((self.num_position,20))
            entropy = torch.zeros(self.num_position)
            for p in range(self.num_position):
                n = self.mask[p].sum()
                eff = np.log(20)/(np.log(n))

                prob = q[p][self.mask[p].bool()]
                entropy[p] = eff * (0-prob*torch.log(prob)).sum()

                for i in range(20):
                    if self.mask[p, i] == 0:
                        continue
                    delta = torch.zeros(20)
                    delta[i] = 1
                    dh[p,i] += ((1+torch.log(q[p]+eps))*q[p]*(delta-q[p,i].repeat(20))).sum()*eff

            dh = dh * lam_arr.unsqueeze(1).repeat(1,20)
            
            dphi -= dh
            
            loss_entropy = (lam_arr * entropy).sum()
            loss = mean_fx + loss_entropy
            
            phi = phi + self.lr * dphi

            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Fitness/train', mean_fx, step)
            writer.add_scalar('Entropy/train', (loss_entropy/lam).item(), step)

        if not os.path.exists(f'results/{self.protein}'):
            os.makedirs(f'results/{self.protein}')
        if not os.path.exists(f'results/{self.protein}/libraries'):
            os.makedirs(f'results/{self.protein}/libraries')

        if self.masked_AAs:
            torch.save(phi, f'results/{self.protein}/libraries/default_{self.seed}_{lam}_{"".join(self.masked_AAs)}.pt')
        else:
            torch.save(phi, f'results/{self.protein}/libraries/default_{self.seed}_{lam}.pt')


    def load_results_default(self, lam_list, parallel=True, num_proc=60):

        path = f'results/{self.protein}/libraries'

        entropy_list, mean_list = [],[]
        indices = range(20**self.num_position)

        for lam in tqdm(lam_list, desc='loading results for MODIFY (default)'):
            
            if self.masked_AAs:
                phi = torch.load(os.path.join(path, f'default_{self.seed}_{lam}_{"".join(self.masked_AAs)}.pt'))
            else:
                phi = torch.load(os.path.join(path, f'default_{self.seed}_{lam}.pt'))
            q = softmax_mask(phi, self.mask)
            
            if parallel:
                with Pool(num_proc, initializer=init_worker_probability, initargs=(q,self.num_position,)) as pool:
                    prob = list(pool.map(calculate_probability, indices, chunksize=1000))
                prob = np.array(prob)
            else:
                prob = []
                for ind in indices:
                    prob.append(calculate_probability(ind))
                prob = np.array(prob)
            
            fitness_prob = self.fitnesses * prob
            mean_fitness = fitness_prob.sum()
            
            entropy = torch.zeros(self.num_position)
            for p in range(self.num_position):
                n = self.mask[p].sum()
                eff = np.log(20)/(np.log(n))
                prob = q[p][self.mask[p].bool()]
                entropy[p] = eff * (0-prob*torch.log(prob)).sum()
            entropy = entropy.mean().item()

            entropy_list.append(entropy)
            mean_list.append(mean_fitness)

        
        res_df = pd.DataFrame({'lambda':lam_list, 
                            'entropy':entropy_list, 
                            'mean fitness':mean_list}).sort_values('lambda')
        
        if self.masked_AAs:
            res_df.to_csv(f'results/{self.protein}/default_pareto_{self.seed}_{"".join(self.masked_AAs)}.csv')
        else:
            res_df.to_csv(f'results/{self.protein}/default_pareto_{self.seed}.csv')


    def optimization_informed(self, resets=[], parallel=True, num_proc=60):

        # Find the default library with maximum fitness*diversity
        if self.masked_AAs:
            df = pd.read_csv(f'results/{self.protein}/default_pareto_{self.seed}_{"".join(self.masked_AAs)}.csv', index_col=0)
        else:
            df = pd.read_csv(f'results/{self.protein}/default_pareto_{self.seed}.csv', index_col=0)
        df['area'] = df.entropy * df['mean fitness']
        entropy_opt, mean_fitness_opt = df.loc[[df.area.argmax()]][['entropy', 'mean fitness']].values.tolist()[0]
        lam = df.loc[[df.area.argmax()]]['lambda'].values[0]
        print('MODIFY-default', lam, entropy_opt, mean_fitness_opt)
        self.lam_default = lam

        lam = (torch.ones(1)*self.lam_default).repeat(self.num_position)

        # informed setting reset lam
        print(resets)
        map_p2i = {j:i for i,j in enumerate(self.positions)}
        for reset in resets:
            lam[map_p2i[int(reset[0])]] = float(reset[1])
        print(lam)

        lam_name = '-'.join(map(str, [round(l, 2) for l in lam.tolist()]))
        print(lam_name)
        self.lam_name = lam_name

        writer = SummaryWriter(f'log/{self.protein}/runs_informed/{self.protein}_{self.seed}_{lam_name}')

        '''Initialize phi'''
        np.random.seed(self.seed)
        random_array = np.random.rand(self.num_position, 20)
        phi = torch.tensor(random_array, dtype=torch.float32)

        eps = 1e-45

        for step in range(self.T):

            ''''Softmax with mask'''
            q = softmax_mask(phi, self.mask)

            '''sample'''
            samples = torch.vstack([torch.multinomial(q[i], self.B, replacement=True) for i in range(self.num_position)]) 
            indices = (samples * 20**(self.num_position-1-torch.arange(self.num_position).view(self.num_position,1).repeat(1,self.B))).sum(dim=0).tolist()
            
            dphi = torch.zeros((self.num_position,20))
            mean_fx = 0
            for i,x in enumerate(indices):
                
                fx = self.fitnesses[x]
                mean_fx += fx

                wx = fx

                delta = torch.zeros((self.num_position,20))
                delta[range(self.num_position),samples[:,i]] = 1

                dphi += wx * (delta - q)

            dphi /= self.B
            mean_fx /= self.B

            
            dh = torch.zeros((self.num_position,20))
            entropy = torch.zeros(self.num_position)
            for p in range(self.num_position):
                n = self.mask[p].sum()
                eff = np.log(20)/(np.log(n))

                prob = q[p][self.mask[p].bool()]
                entropy[p] = eff * (0-prob*torch.log(prob)).sum()

                for i in range(20):
                    if self.mask[p, i] == 0:
                        continue
                    delta = torch.zeros(20)
                    delta[i] = 1
                    dh[p,i] += ((1+torch.log(q[p]+eps))*q[p]*(delta-q[p,i].repeat(20))).sum()*eff

            dh = dh * lam.unsqueeze(1).repeat(1,20)
            
            dphi -= dh
            
            loss_entropy = (lam * entropy).sum()
            loss = mean_fx + loss_entropy
            
            phi = phi + self.lr * dphi

            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Fitness/train', mean_fx, step)
            writer.add_scalar('Entropy/train', (loss_entropy/lam[0]).item(), step)

        if self.masked_AAs:
            torch.save(phi, f'results/{self.protein}/libraries/informed_{self.seed}_{lam_name}_{"".join(self.masked_AAs)}.pt')
        else:
            torch.save(phi, f'results/{self.protein}/libraries/informed_{self.seed}_{lam_name}.pt')

        
        # calculate entropy and fitness for MODIFY-informed
        q = softmax_mask(phi, self.mask)
        indices = range(20**self.num_position)

        if parallel:
            with Pool(num_proc, initializer=init_worker_probability, initargs=(q,self.num_position,)) as pool:
                prob = list(pool.map(calculate_probability, indices, chunksize=1000))
            prob = np.array(prob)
        else:
            prob = []
            for ind in indices:
                prob.append(calculate_probability(ind))
            prob = np.array(prob)
        
        fitness_prob = self.fitnesses * prob
        mean_fitness = fitness_prob.sum()
        
        entropy = torch.zeros(self.num_position)
        for p in range(self.num_position):
            n = self.mask[p].sum()
            eff = np.log(20)/(np.log(n))
            prob = q[p][self.mask[p].bool()]
            entropy[p] = eff * (0-prob*torch.log(prob)).sum()
        entropy = entropy.mean().item()

        entropy_final, mean_fitness_final = entropy, mean_fitness

        res_df = pd.DataFrame({'lambda':[lam_name], 
                            'entropy':[entropy_final], 
                            'mean fitness':[mean_fitness_final]}).sort_values('lambda')
        
        if self.masked_AAs:
            res_df.to_csv(f'results/{self.protein}/informed_pareto_{self.seed}_{"".join(self.masked_AAs)}_{lam_name}.csv')
        else:
            res_df.to_csv(f'results/{self.protein}/informed_pareto_{self.seed}_{lam_name}.csv')
