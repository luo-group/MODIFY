import argparse
import sys
sys.path.append('.')

from modify.optimization import MODIFYOptimization


def main(args):
    
    if not args.informed:
        '''default setting of MODIFY'''

        positions = list(map(int, args.positions.split(',')))
        masked_AAs = args.masked_AAs.split(',') if args.masked_AAs!='' else []

        modify_opt = MODIFYOptimization(
            protein=args.protein,
            offset=args.offset,
            positions=positions,
            masked_AAs=masked_AAs,
            fitness_col=args.fitness_col,
            seed=args.seed,
            lr=args.lr,
            B=args.B,
            T=args.T,
        )

        lam_list = [round(i*0.01, 2) for i in range(1,201)]

        modify_opt.parallel_optimization_default(
            lam_list=lam_list,
            parallel=args.parallel,
            num_proc=args.num_proc)
        modify_opt.load_results_default(
            lam_list=lam_list,
            parallel=args.parallel,
            num_proc=args.num_proc)

    
    else:
        '''informed setting of MODIFY'''
    
        positions = list(map(int, args.positions.split(',')))
        masked_AAs = args.masked_AAs.split(',') if args.masked_AAs!='' else []

        modify_opt = MODIFYOptimization(
            protein=args.protein,
            offset=args.offset,
            positions=positions,
            masked_AAs=masked_AAs,
            fitness_col=args.fitness_col,
            seed=args.seed,
            lr=args.lr,
            B=args.B,
            T=args.T,
        )

        resets = [a.split('-') for a in args.resets]

        modify_opt.optimization_informed(
            resets=resets,
            parallel=args.parallel,
            num_proc=args.num_proc
        )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run library design using MODIFY')
    parser.add_argument('--protein', type=str, default='GB1', help='Target protein name')
    parser.add_argument('--offset', type=int, default=1, help='Index for the first amino acid in the wildtype sequence')
    parser.add_argument('--positions', type=str, default='39,40,41,54', help='Target residues connected by comma (e.g., \"39,40,41,54\")')
    parser.add_argument('--masked_AAs', type=str, default='', help='Masked AAs connected by comma (e.g., \"39L,41G\")')
    parser.add_argument('--fitness_col', type=str, default='modify_fitness', help='Name of column storing zero-shot predicted fitness values')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing with multiple cpu processes')
    parser.add_argument('--num_proc', type=int, default=60, help='Number of cpu processes')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--B', type=int, default=1000, help='Batch size')
    parser.add_argument('--T', type=int, default=2000, help='Total number of steps')
    parser.add_argument('--informed', action='store_true', help='To use informed setting')
    parser.add_argument('--resets', nargs='+',help='lambda adjustment for informed setting (e.g., 40-0.69)')
    args = parser.parse_args()

    main(args)
