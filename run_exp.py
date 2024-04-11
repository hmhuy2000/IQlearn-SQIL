import multiprocessing
import subprocess
import sys
import argparse

def run_script(args):
    (seed) = args
    print(seed)

    # subprocess.run([
    #         'python', '-u' ,'train_iq.py', f'env=cheetah', 'agent=sac',
    #         f'expert.demos=1000', 'method.loss=value', 'method.regularize=True',
    #         'agent.actor_lr=3e-05', 
    #         'env.eval_interval=1e4', 'env.learn_steps=1e6', f'seed={seed}',
    #     ])

    # subprocess.run([
    #         'python', '-u' ,'train_iq.py', f'env=ant', 'agent=sac',
    #         f'expert.demos=1000', 'method.loss=value', 'method.regularize=True',
    #         'agent.actor_lr=3e-05', 'agent.init_temp=0.001',
    #         'env.eval_interval=1e4', 'env.learn_steps=1e6', f'seed={seed}',
    #     ])
    
    subprocess.run([
            'python', '-u' ,'train_iq.py', f'env=walker', 'agent=sac',
            f'expert.demos=1000', 'method.loss=v0', 'method.regularize=True',
            'agent.actor_lr=3e-05',
            'env.eval_interval=1e4', 'env.learn_steps=1e6', f'seed={seed}',
        ])

    # subprocess.run([
    #         'python', '-u' ,'train_iq.py', f'env=hopper', 'agent=sac',
    #         f'expert.demos=1000', 'method.loss=v0', 'method.regularize=True',
    #         'agent.actor_lr=3e-05',
    #         'env.eval_interval=1e4', 'env.learn_steps=1e6', f'seed={seed}',
    #     ])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run scripts concurrently.')
    parser.add_argument('--max_processes', type=int, default=5, help='Number of scripts to run concurrently')
    parser.add_argument('--seed_list',default='1,2', help='seed list for experiments')
    args = parser.parse_args()
    run_pools = [(int(seed)) for seed in args.seed_list.split(',')]

    with multiprocessing.Pool(processes=args.max_processes) as pool:
        pool.map(run_script, run_pools)
    print("All scripts have finished running.")