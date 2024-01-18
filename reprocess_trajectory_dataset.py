import numpy as np
from tqdm import tqdm

save_path = '/home/alison/Clay_Data/Fully_Processed/Jan17_Human_Demos'
traj_path = '/home/alison/Clay_Data/Trajectory_Data/Aug_Dec14_Human_Demos/X'

# the number of actions for each of the base trajectories
trajectory_dict = {0: 3,
                   1: 6,
                   2: 7,
                   3: 7,
                   4: 7,
                   5: 7,
                   6: 7,
                   7: 8,
                   8: 7,
                   9: 7}

action_list = []

traj_idx = 0
save_idx = 0
# iterate through the base trajectories
for i in tqdm(range(10)):
    # iterate through the rotation augmentations
    for j in tqdm(range(90)):
        n_actions = trajectory_dict[i]

        traj_folder = traj_path + '/Trajectory' + str(traj_idx)

        # iterate through the actions 
        for k in range(n_actions):
            # load action
            a = np.load(traj_folder + '/action' + str(k) + '.npy')
            action_list.append(a)

            # load state
            s = np.load(traj_folder + '/pointcloud' + str(k) + '.npy')
            np.save(save_path + '/States/state' + str(save_idx) + '.npy', s)

            # load next state
            ns = np.load(traj_folder + '/pointcloud' + str(k+1) + '.npy')
            np.save(save_path + '/Next_States/next_state' + str(save_idx) + '.npy', ns)

            save_idx += 1

        traj_idx += 1

np.save(save_path + '/action_normalized.npy', action_list)
