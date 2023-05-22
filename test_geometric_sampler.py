import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from models.dvae import *

def __calculate_dist(self, current, target):
    '''
    This function is used to calculate the distance between each point
    input: 
        current -- the 3d position of the point in point cloud, shape is [num_points, 3]
        target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    output: 
        dist  -- the matrix indicating the distance, the shape is [num_points, num_points] 
    '''

    assert current.shape[0] == target.shape[0] # make sure the number of points is the same
    dist = cdist(current,target,metric= 'euclidean')
    return dist
    
def __match_point_pairs(self, current, target):
    '''
    This function is used to match the points with their nearest neighbor
    input:
        current -- the 3d position of the point in point cloud, shape is [num_points, 3]
        target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    output:
        indice -- the array of the nearest neighbor's index for current points
        distance -- the array of the nearest neighbor's distance for current points
    ''' 
    dist = self.__calculate_dist(current, target)
    indice = np.argmin(dist,axis=1)
    distance = np.min(dist,axis=1)
    return indice,distance

# def __get_rotation_from_vector(self,vector):
#     '''
#     This function is used to calculate the rotation of a vector. Since we will get a vector (a line in 3D space) from two points, we need to transfer the line into rotation. 
#     input:
#         vector -- the vector between two points, the shape is [1,3]
#     ourput:
#         rotation -- the rotation in 3d space [roll, pitch, yaw]
#         ######## Attention ######## 
#         Here might be some +/- problems. If the final action is not corresponding to our target, maybe debugging here would solve the problem.
#     '''
#     normal_vector = np.array([1,0,0])
#     quaternion = u.get_quaternion_from_vectors(vector,normal_vector)
#     rotation = u.get_euler_from_quaternion(quaternion)
#     return rotation

def plan_a(self, current, target):
    '''
    Here is the implement of Strategy A
        1. Match the point with its neareat point;
        2. The distance between them is the sample weight;
        3. Random sample two points, take the center of them as the picking position, and the distance for the gripper closing.
    input:
        current -- the 3d position of the point in point cloud, shape is [num_points, 3]
        target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    output:
        action -- list, the action that random sampled, the dimension is [x,y,z,roll,pitch,yaw,d]
    '''
    idx,dist = self.__match_point_pairs(current, target)
    point_a,point_b = np.random.choice(np.arange(idx.shape[0]),2,p=dist/np.sum(dist))
    pos_a = current[point_a,:]
    pos_b = current[point_b,:]
    vector = pos_a - pos_b
    distance = np.sqrt(np.sum(vector**2))
    # rotation = self.__get_rotation_from_vector(vector)
    # action = list((pos_a + pos_b)/2) + list(rotation) + [distance]

    # need to take the x,y,z and divide by 10 add the center of the clay to it (point cloud is normalized and scaled???)
    # need to only take distances within the gripper's limits
    # return action
    return idx, dist, point_a, point_b, pos_a, pos_b, vector, dist #, rotation, action

def plan_b(self, current, target):
    '''
    Here is the implement of Strategy B
        1. Match the point with its neareat point;
        2. Take the top two points with largest distance;
        3. Get the vector of these two points; 
        4. Around this vector, randomly sample actions.
    input:
        current -- the 3d position of the point in point cloud, shape is [num_points, 3]
        target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    output:
        action -- list, the action that random sampled, the dimension is [x,y,z,roll,pitch,yaw,d]
    '''
    DISTANCE = 0.1 # the range of gripper closing
    NOISY = 0.01 # the range of noisy for random smaple

    idx,dist = self.__match_point_pairs(current, target)
    point_a, point_b = idx[np.argpartition(dist,-2)[-2:]]
    pos_a = current[point_a,:]
    pos_b = current[point_b,:]
    vector = pos_a - pos_b
    vector += np.random.rand(3) * NOISY
    # rotation = self.__get_rotation_from_vector(vector)
    distance = np.random.rand() * DISTANCE
    # action = list((pos_a + pos_b)/2) + list(rotation) + [distance]
    # return action


# load a target pcl
target = np.load('target_clouds/X.npy')

# load a current pcl
state = np.load('/home/alison/Clay_Data/Fully_Processed/May4_5D/States/shell_scaled_state0.npy')

state = torch.from_numpy(state)
state = torch.unsqueeze(state, 0).to(torch.float32)
state = state.cuda()
target = torch.from_numpy(target)
target = torch.unsqueeze(target, 0).to(torch.float32)
target = target.cuda()

# group = Group(num_group=32, group_size=64)
group = Group(num_group=64, group_size=32)
# group = Group(num_group=128, group_size=16)
t_neigh, t_ctr = group(target)
s_neigh, s_ctr = group(state)

# ------ visualize the centers of the state and target -------
state_centers = s_ctr.squeeze().detach().cpu().numpy()
target_centers = t_ctr.squeeze().detach().cpu().numpy()

og_pcl = o3d.geometry.PointCloud()
og_pcl.points = o3d.utility.Vector3dVector(target_centers)
og_colors = np.tile(np.array([0, 1, 0]), (len(target_centers),1))
og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(state_centers)
pcl_colors = np.tile(np.array([1, 0, 0]), (len(state_centers),1))
pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
# o3d.visualization.draw_geometries([pcl, og_pcl])

# ------- determine the closes point pairs -------
state_tree = KDTree(target_centers)
dists, idxs = state_tree.query(state_centers)
distances = dists.reshape(-1,1)
print("distances shape: ", distances.shape)
closest_pairs = np.column_stack((state_centers, target_centers[idxs]))
closest_pairs = np.concatenate((closest_pairs, distances), axis=1)
print("closest pairs shape: ", closest_pairs.shape)
sorted_pairs = closest_pairs[(-distances).argsort()]
print("sorted pairs shape: ", sorted_pairs.shape)
# assert False

# print("closest pairs: ", closest_pairs)
# print("\nClosest pairs shape: ", closest_pairs.shape)

# dists = square_distance(s_ctr, t_ctr) # dist shape [1, s_pts, t_pts]
# idxs = torch.argmin(torch.squeeze(dists, 0), dim=1) # indices of t_ctr that are the closest to s_ctr
# s_idxs = torch.linspace(0, dists.size()[1] - 1, dists.size()[1]).to(torch.int32)
# print("\ns idxs: ", s_idxs)
# distances = torch.squeeze(dists, 0)[s_idxs.long(), idxs.long()]

# idx = idxs.detach().cpu().numpy()
# dst = distances.detach().cpu().numpy()
# combined = zip(idx,dst)
# sort = sorted(combined, key=lambda x: x[1], reverse=True)
# idx = [x[0] for x in sort]
# dst = [x[1] for x in sort]

# for i in range(len(idx)):
#     state_pt = state_centers[i,:]
#     target_pt = target_centers[idx[i],:]
#     pair = np.vstack((np.expand_dims(state_pt, axis=0), np.expand_dims(target_pt, axis=0)))

#     print("dst: ", dst[i])

for i in range(len(sorted_pairs)):
    state_pt = sorted_pairs[i,0,0:3]
    target_pt = sorted_pairs[i,0,3:6]
    pair = np.vstack((np.expand_dims(state_pt, axis=0), np.expand_dims(target_pt, axis=0)))

    print("Pair: ", pair)

    point_pair = o3d.geometry.PointCloud()
    point_pair.points = o3d.utility.Vector3dVector(pair)
    pair_colors = np.tile(np.array([0, 0, 1]), (len(pair),1))
    point_pair.colors = o3d.utility.Vector3dVector(pair_colors)
    o3d.visualization.draw_geometries([pcl, og_pcl, point_pair])

# ------- alternate planner -------
# do a clustering of the cloud (farthest point + knn) similar to what's done with dvae

# do the matching point pairs with the centroid of the clusters of the state and target

# look for the biggest difference between the state & target (target needs to be closer to center
# than state)

# select the 4 points that lay on the line???? (start without this)

# select action within this region

# alternatively select candidate actions with the N largest distances and pass through dynamics to select
# the best one?

# pass action through dynamics model - predict next state and re-plan???

