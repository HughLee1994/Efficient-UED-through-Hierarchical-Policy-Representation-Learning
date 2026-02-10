import argparse


parser = argparse.ArgumentParser(description='New MDP with Diffusion Model')

# Basic setup arguments.
parser.add_argument(
    '--actor_lr',
    type=float,
    default=1e-3,
    help='Learning rate of actor'
)
parser.add_argument(
    '--critic_lr',
    type=float,
    default=3e-3,
    help='Learning rate of critic'
)
parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discount parameter'
)
parser.add_argument(
    '--lam',
    type=float,
    default=0.95,
    help='Hyperparameter for GAE'
)
parser.add_argument(
    '--ent_coef',
    type=float,
    default=0.01,
    help='coefficient for the entropy bonus (to encourage exploration)'
)
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='Experiment random seed.')
parser.add_argument(
    '--domain_randomization',
    type=eval,
    default=False,
    help='Use domain randomization to generate environments')
parser.add_argument(
    '--accel',
    type=eval,
    default=False,
    help='human level curriculum by ACCEL algorithm')
parser.add_argument(
    '--newMDP',
    type=eval,
    default=True,
    help='formulate the problem as hierarchical MDP structure')
parser.add_argument(
    '--cv',
    type=eval,
    default=False,
    help='whether to use coefficient of variation as the cv reward in the reward function')
parser.add_argument(
    '--coef_cv',
    type=float,
    default=1e-2,
    help='coefficient to balance the trade-off between the mean reward and the coefficient of variation reward'
)
parser.add_argument(
    '--batch_size',
    type=int,
    # default=10,     # for testing
    default=32,
    help='coefficient to balance the trade-off between the mean reward and the coefficient of variation reward'
)
parser.add_argument(
    '--iterations',
    type=int,
    # default=5,      # for testing
    default=20,
    help='iterations to train the teacher agent for one batch of experience'
)


# Environment setting
parser.add_argument(
    '--num_env',
    type=int,
    default=5,
    # default=10,  # for testing
    help='number of environments training in parallel.')
parser.add_argument(
    '--num_udpates_per_env',
    type=int,
    # default=1,  # for testing
    default=20,
    help='number of update per environment.')
parser.add_argument(
    '--num_budget',
    type=int,
    default=500,
    help='number of budget can be used to generate environments.')
parser.add_argument(
    '--num_steps_per_update',
    type=int,
    default=128,
    help='number of time steps in each run during one update.')
parser.add_argument(
    '--num_eval_envs',
    type=int,
    default=10,
    help='number of test environments to evaluate transfer performance in parallel.')
parser.add_argument(
    '--teacher_max_action',
    type=float,
    default=[-9.0, 12.00, 1.50],
    help='maximum action space'
)
parser.add_argument(
    '--buffer_size',
    type=int,
    default=10000,
    help='size of the teacher agent replay buffer')

# saving setting
parser.add_argument(
    '--log_dir',
    default='./logs',
    help='Directory in which to save experimental outputs.')
parser.add_argument(
    '--save_weights',
    type=eval,
    default=True,
    help='save weights')
parser.add_argument(
    '--load_weights',
    type=eval,
    default=True,
    help='load weights')
parser.add_argument(
    '--actor_weights_path',
    default='actor_weights.h5',
    help='Directory in which to save actor weights outputs.')
parser.add_argument(
    '--critic_weights_path',
    default='critic_weights.h5',
    help='Directory in which to save critic weights outputs.')
parser.add_argument(
    '--cuda',
    default=True,
    help='Using CUDA training.')


# Diffusion model arguments
parser.add_argument('--use_diffusion', default=True, type=eval, help='Combine the diffusion model in training the teacher agent')
parser.add_argument('--diffusion_discount', default=0.99, type=float)

parser.add_argument('--diffusion_num_steps_per_epoch', default=50, type=int)
parser.add_argument('--diffusion_num_epochs', 
                    # default=1,      # for testing 
                    default=5,
                    type=int)
parser.add_argument('--diffusion_eval_freq', default=50, type=int)

parser.add_argument('--diffusion_batch_size',
                    default=64,
                    # default=10,     # for testing
                    type=int)
parser.add_argument('--diffusion_lr_decay', action='store_true')

# RL parameters
parser.add_argument('--diffusion_tau', default=0.005, type=float)

# Diffusion setting
parser.add_argument('--diffusion_T', default=5, type=int)
parser.add_argument('--diffusion_beta_schedule', default='vp', type=str)

parser.add_argument('--diffusion_lr', default=3e-4, type=float)
parser.add_argument('--diffusion_seed', default=1, type=int)

parser.add_argument('--diffusion_eval_per_epoch',
                    default=10,
                    # default=3,      # for test
                    type=int)
parser.add_argument('--diffusion_size_use_syn_buffer',
                    default=500,
                    # default=5,      # for test
                    type=int,
                    help='the size of synthetic experience buffer when start to use it to train the teacher agent')
parser.add_argument('--diffusion_syn_ratio',
                    # default=10,
                    default=0.25,      # for test
                    type=float,
                    help='the ratio of synthetic experience buffer when using it to train the teacher agent')
parser.add_argument('--diffusion_max_state', default=300, type=int, help='maximum state')
parser.add_argument('--diffusion_synth_buffer_size', default=2000, type=int, help='the synthetic experience buffer generated by diffusion model')

