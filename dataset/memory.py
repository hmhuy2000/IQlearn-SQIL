from collections import deque
import numpy as np
import random
import torch

from dataset.expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.seed = seed
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        # If path has no extension add npy
        if not (path.endswith("pkl") or path.endswith("hdf5")):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        self.full_trajs = data.full_trajectories
        
        self.memory_size = data.__len__()
        self.buffer = deque(maxlen=self.memory_size)
        obs_arr = []
        next_obs_arr = []
        for i in range(len(data)):
            self.add(data[i])
            obs_arr.append(data[i][0])
            next_obs_arr.append(data[i][1])
        return obs_arr,next_obs_arr

    def process_data(self,dataset,data_length):
        starts_done = np.where(np.array(dataset['terminals'])>0)[0].tolist()
        starts_timeout = np.where(np.array(dataset['timeouts'])>0)[0].tolist()
        starts = [-1]+starts_timeout+starts_done
        starts = list(dict.fromkeys(starts))
        starts.sort()
        rng = np.random.RandomState(self.seed)
        perm = np.arange(len(starts)-1)
        perm = rng.permutation(perm)
        total_length = 0
        for num_traj in range(len(perm)):
            traj_len = (starts[perm[num_traj]+1]+1) - (starts[perm[num_traj]]+1)
            total_length += traj_len
            if (total_length>=data_length):
                break
        num_traj += 1
        idx = perm[:num_traj]
        trajs = {}
        
        trajs['dones'] = [np.array(dataset['terminals'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['states'] = [np.array(dataset['observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['initial_states'] = np.array([dataset['observations'][starts[idx[i]]+1]
                            for i in range(len(idx))])
        trajs['next_states'] = [np.array(dataset['next_observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['actions'] = [np.array(dataset['actions'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['rewards'] = [dataset['rewards'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                                for i in range(len(idx))]
            
        trajs['dones'] = np.concatenate(trajs['dones'],axis=0)[:data_length]
        trajs['states'] = np.concatenate(trajs['states'],axis=0)[:data_length]
        trajs['actions'] = np.concatenate(trajs['actions'],axis=0)[:data_length]
        trajs['next_states'] = np.concatenate(trajs['next_states'],axis=0)[:data_length]
        reward_arr = [np.sum(trajs['rewards'][i]) for i in range(len(trajs['rewards']))]
              
        trajs['rewards'] = np.concatenate(trajs['rewards'],axis=0)[:data_length]
        print(f'{len(idx)}/{len(perm)} trajectories')
        print(f'return: {np.mean(reward_arr):.2f} ± {np.std(reward_arr):.2f}, Q1 = {np.percentile(reward_arr, 25):.2f}'+
            f', Q2 = {np.percentile(reward_arr, 50):.2f}, Q3 = {np.percentile(reward_arr, 75):.2f},'+
            f' min = {np.min(reward_arr):.2f}, max = {np.max(reward_arr):.2f}')
        return trajs

    def load_from_dataset(self, dataset, num_trajs):
        dataset = self.process_data(dataset,num_trajs)
        self.memory_size = len(dataset['states'])
        self.buffer = deque(maxlen=self.memory_size)
        for i in range(len(dataset['states'])):
            self.add((
                dataset['states'][i],
                dataset['next_states'][i],
                dataset['actions'][i],
                dataset['rewards'][i],
                dataset['dones'][i]
            ))


    def process_maze_data(self,dataset,data_length):
        starts_done = np.where(np.array(dataset['terminals'])>0)[0].tolist()
        starts = [-1]+starts_done
        starts = list(dict.fromkeys(starts))
        starts.sort()
        rng = np.random.RandomState(self.seed)
        perm = np.arange(len(starts)-1)
        perm = rng.permutation(perm)
        total_length = 0
        for num_traj in range(len(perm)):
            traj_len = (starts[perm[num_traj]+1]+1) - (starts[perm[num_traj]]+1)
            total_length += traj_len
            if (total_length>=data_length):
                break
        num_traj += 1
        idx = perm[:num_traj]
        trajs = {}
        
        trajs['dones'] = [np.array(dataset['terminals'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['states'] = [np.array(dataset['observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['initial_states'] = np.array([dataset['observations'][starts[idx[i]]+1]
                            for i in range(len(idx))])
        trajs['next_states'] = [np.array(dataset['next_observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['actions'] = [np.array(dataset['actions'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                            for i in range(len(idx))]
        trajs['rewards'] = [dataset['rewards'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                                for i in range(len(idx))]
            
        trajs['dones'] = np.concatenate(trajs['dones'],axis=0)[:data_length]
        trajs['states'] = np.concatenate(trajs['states'],axis=0)[:data_length]
        trajs['actions'] = np.concatenate(trajs['actions'],axis=0)[:data_length]
        trajs['next_states'] = np.concatenate(trajs['next_states'],axis=0)[:data_length]
        reward_arr = [np.sum(trajs['rewards'][i]) for i in range(len(trajs['rewards']))]
              
        trajs['rewards'] = np.concatenate(trajs['rewards'],axis=0)[:data_length]
        print(f'{len(idx)}/{len(perm)} trajectories')
        print(f'return: {np.mean(reward_arr):.2f} ± {np.std(reward_arr):.2f}, Q1 = {np.percentile(reward_arr, 25):.2f}'+
            f', Q2 = {np.percentile(reward_arr, 50):.2f}, Q3 = {np.percentile(reward_arr, 75):.2f},'+
            f' min = {np.min(reward_arr):.2f}, max = {np.max(reward_arr):.2f}')
        return trajs

    def load_from_maze_dataset(self, dataset, num_trajs):
        dataset = self.process_maze_data(dataset,num_trajs)
        self.memory_size = len(dataset['states'])
        self.buffer = deque(maxlen=self.memory_size)
        for i in range(len(dataset['states'])):
            self.add((
                dataset['states'][i],
                dataset['next_states'][i],
                dataset['actions'][i],
                dataset['rewards'][i],
                dataset['dones'][i]
            ))


    def get_samples(self, batch_size, device):
        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(
            *batch)

        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)
        
        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done


class StateBuffer(object):
    def __init__(self, state_dim, buffer_size, seed):
        self.state_dim = state_dim
        self.buffer_size =  int(buffer_size)
        self.expert_buffer = []
        self.rollout_buffer = []
        self.index = 0
        random.seed(seed)

    def add_expert(self, state,score):
        self.expert_buffer.append((state,score))

    def add_rollout(self, state,score):
        if (len(self.rollout_buffer) < self.buffer_size):
            self.rollout_buffer.append((state,score))
            self.index = (self.index+1)%self.buffer_size
        else:
            self.rollout_buffer[self.index] = (state,score)
            self.index = (self.index+1)%self.buffer_size

    def get_expert(self, batch_size):
        if len(self.expert_buffer) < batch_size:
            batch_size = len(self.expert_buffer)
        indexes = np.random.choice(np.arange(len(self.expert_buffer)), size=batch_size, replace=False)
        return [self.expert_buffer[i] for i in indexes]
    
    def get_rollout(self, batch_size):
        if len(self.rollout_buffer) < batch_size:
            batch_size = len(self.rollout_buffer)
        indexes = np.random.choice(np.arange(len(self.rollout_buffer)), size=batch_size, replace=False)
        return [self.rollout_buffer[i] for i in indexes]

    def get_samples(self, batch_size,device):
        expert_batch = self.get_expert(batch_size//2)
        rollout_batch = self.get_rollout(batch_size//2)
        batch = expert_batch + rollout_batch
        (_,exp_score) = zip(*expert_batch)
        (_,pi_score) = zip(*rollout_batch)
        exp_score = torch.tensor(np.array(exp_score), dtype=torch.float, device=device)
        pi_score = torch.tensor(np.array(pi_score), dtype=torch.float, device=device)
        
        is_expert = torch.cat([torch.ones_like(exp_score, dtype=torch.bool),
                           torch.zeros_like(pi_score, dtype=torch.bool)], dim=0)
        (state, score) = zip(*batch)
        state = torch.as_tensor(np.array(state), dtype=torch.float, device=device)
        score = torch.as_tensor(np.array(score), dtype=torch.float, device=device).unsqueeze(1)
        return state, score, is_expert