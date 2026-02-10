import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import pickle

# from utils.data_sampler import Data_Sampler
# from utils.data_sampler import ReplayMemory, gen_dataset
from util.utils import EarlyStopping, reward_fn
from agent import Agent_Diffusion


# fn = 'dataset.pkl'
#
# with open(fn, 'rb') as f:
#     Data_Sampler = pickle.load(f)
# Data_Sampler = gen_dataset()
# print(type(Data_Sampler))

def train_diffusion_model(agent, data, synth_buffer, state_dim, action_dim, max_action, max_state, device, args, writer):
    data_sampler = data

    # agent = Agent(state_dim=state_dim,
    #               action_dim=action_dim,
    #               max_action=max_action,
    #               max_state=max_state,
    #               device=device,
    #               discount=args.diffusion_discount,
    #               tau=args.diffusion_tau,
    #               beta_schedule=args.diffusion_beta_schedule,
    #               n_timesteps=args.diffusion_T,
    #               lr=args.diffusion_lr)

    early_stop = False
    stop_check = EarlyStopping(tolerance=1, min_delta=0.)

    # SummaryWriter
    # writer = SummaryWriter()
    log_dir_diffusion_model = "./diffusion_models"
    # log_dir = "./logs"
    if not os.path.exists(log_dir_diffusion_model):
        os.makedirs(log_dir_diffusion_model)
    # writer = SummaryWriter(log_dir)
    evaluation = []

    training_iters = 0
    max_timesteps = args.diffusion_num_epochs * args.diffusion_num_steps_per_epoch
    metric = 100.
    step = 0
    training_iter = 0

    while training_iter < args.diffusion_num_epochs:
        # iterations = int(args.diffusion_eval_freq * args.diffusion_num_steps_per_epoch)
        iterations = args.diffusion_num_steps_per_epoch
        loss_metric = agent.train(data_sampler, iterations=iterations,
                                  batch_size=args.diffusion_batch_size,
                                  log_writer=writer)
        training_iters += iterations
        training_iter += 1
        step += 1
        # print("@@@@@@@@@@@ training epoch", step)

        curr_epoch = int(training_iters // int(args.diffusion_num_steps_per_epoch))

        # writer logging for diffusion
        # writer.add_scalar('Trained Epochs', curr_epoch)
        # writer.add_scalar('BC Loss', np.mean(loss_metric['bc_loss']), step)
        # # writer.add_scalar('QL Loss', np.mean(loss_metric['ql_loss']), step)
        # writer.add_scalar('Actor Loss', np.mean(loss_metric['actor_loss']), step)
        # writer.add_scalar('Critic Loss', np.mean(loss_metric['critic_loss']), step)

        # bc_loss = np.mean(loss_metric['bc_loss'])
        # metric = bc_loss
    for i in range(args.diffusion_eval_per_epoch):
        states, actions, next_states, rewards, done = eval_policy(data_sampler, agent, batch_size=1, eval_episodes=1000)
        # states = states.squeeze()
        # actions = actions.squeeze()
        # next_states = next_states.squeeze()
        # rewards = rewards.squeeze()
        # done = done.squeeze()
        synth_buffer.add(states, actions, next_states, rewards, done)
        # synth_buffer.buffer.append((states, actions, next_states, rewards, done))
        # evaluation.append((states, actions, next_states, rewards))
        agent.save_model(log_dir_diffusion_model, curr_epoch)

        # np.save(os.path.join(log_dir, "eval"), evaluation)
        fn = 'eval_data.pkl'
        with open(fn, 'wb') as f:
            tmp = pickle.dump(evaluation, f)
    # args.save_model(log_dir, curr_epoch)
    # writer.close()


def eval_policy(dataset, agent, batch_size=1, eval_episodes=1000):
    states = []
    actions = []
    next_states = []
    rewards = []
    # dataset = data
    device = agent.device
    # data_sampler = Data_Sampler(dataset, device, reward_tune='no')
    data_sampler = dataset
    num = 0
    # for _ in range(eval_episodes):
    state, action, next_state, reward, not_done = data_sampler.sample(batch_size)
    # print("@@@@@@@@@@ reward", reward)
    # if not not_done:
    num += 1
    # print("@@@@@@@@@@ not_done times", num)
    action, next_state = agent.sample_action_state(state, batch_size)
    state = state.numpy()[0]
    reward = reward.numpy()[0][0]
    not_done = not_done.numpy()[0][0]
    # action = action.to(torch.float32)
    # next_state = next_state.to(torch.float32)

    # print("@@@@@@@@@@ not_done times", num)
    # print("### debugging next_state", next_state, type(next_state))
    # print("### debugging action", action, type(action))
    # print("### debugging not_done", not_done, type(not_done))
    return state, action, next_state, reward, not_done


# import argparse
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--diffusion_discount', default=0.99, type=float)
#     # parser.add_argument('--device', default='cuda', type=str)
#
#     parser.add_argument('--diffusion_num_steps_per_epoch', default=100, type=int)
#     parser.add_argument('--diffusion_num_epochs', default=1000, type=int)
#     parser.add_argument('--diffusion_eval_freq', default=50, type=int)
#
#     parser.add_argument('--diffusion_batch_size', default=256, type=int)
#     parser.add_argument('--diffusion_lr_decay', action='store_true')
#
#     # RL parameters
#     parser.add_argument('--diffusion_tau', default=0.005, type=float)
#
#     # Diffusion setting
#     parser.add_argument('--diffusion_T', default=5, type=int)
#     parser.add_argument('--diffusion_beta_schedule', default='vp', type=str)
#
#     parser.add_argument('--diffusion_lr', default=3e-4, type=float)
#     parser.add_argument('--diffusion_seed', default=1, type=int)
#     parser.add_argument('--no_cuda', type=bool, nargs='?', const=True, default=True,
#         help='Disables CUDA training.')
#     args = parser.parse_args()
#
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if args.cuda else "cpu")
#     if 'cuda' in device.type:
#         torch.backends.cudnn.benchmark = True
#         print('Using CUDA\n')
#     torch.manual_seed(args.diffusion_seed)
#     np.random.seed(args.diffusion_seed)
#
#     max_state = 20
#     max_action = 8
#     state_dim = 5
#     action_dim = 3
#
#     # fn = 'dataset.pkl'
#     #
#     # with open(fn, 'rb') as f:
#     #     Data_Sampler = pickle.load(f)
#     dataset = gen_dataset(100000)
#     train_diffusion_model(agent, dataset, synth_buffer, state_dim, action_dim, max_action, max_state, device, args, writer)

