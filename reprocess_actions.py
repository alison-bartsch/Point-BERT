import numpy as np

path = '/home/alison/Clay_Data/Fully_Processed/Aug24_Human_Demos_Fully_Processed/Actions'

actions_normalized = []

for i in range(8160):

    # if i % 60 == 0:
    #     a = np.load(path + '/action_5d' + str(i) + '.npy')
    #     actions_normalized.append(a)
    # else:
    a = np.load(path + '/action_5d_normalized_' + str(i) + '.npy')
    actions_normalized.append(a)

np.save('/home/alison/Clay_Data/Fully_Processed/Aug24_Human_Demos_Fully_Processed/action_normalized.npy', actions_normalized)
print("Actions Normalized: ", actions_normalized)