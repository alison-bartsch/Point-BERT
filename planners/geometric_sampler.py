import torch
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
import planner_utils as u
from pointnet2_ops import pointnet2_utils

'''
Here we use two strategies:
    A)  1. Match the point with its neareat point;
        2. The distance between them is the sample weight;
        3. Random sample two points, take the center of them as the picking position, and the distance for the gripper closing.

    B)  1. Match the point with its neareat point;
        2. Take the top two points with largest distance;
        3. Get the vector of these two points; 
        4. Around this vector, randomly sample actions.
'''

DISTANCE = 0.1 # the range of gripper closing
NOISY = 0.01 # the range of noisy for random smaple

class GeometricSampler():

    def __init__(self,args) -> None:
        # self.n_actions = args.n_actions
        pass

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
    
    def __get_rotation_from_vector(self,vector):
        '''
        This function is used to calculate the rotation of a vector. Since we will get a vector (a line in 3D space) from two points, we need to transfer the line into rotation. 
        input:
            vector -- the vector between two points, the shape is [1,3]
        ourput:
            rotation -- the rotation in 3d space [roll, pitch, yaw]
            ######## Attention ######## 
            Here might be some +/- problems. If the final action is not corresponding to our target, maybe debugging here would solve the problem.
        '''
        normal_vector = np.array([1,0,0])
        quaternion = u.get_quaternion_from_vectors(vector,normal_vector)
        rotation = u.get_euler_from_quaternion(quaternion)
        return rotation
    
    def __ctr_dist(self, pcl):
        """
        Given a point cloud, get the distance from (0,0,0) to all of the points in the cloud
        """

        center = np.zeros((1,3)) # np.zeros((pcl.shape[0],3))
        dist = cdist(center,pcl,metric= 'euclidean')
        dist = dist.reshape(-1)
        return dist
    
    def __fps(self, data, number):
        '''
            data B N 3
            number int
        '''
        print("Data: ", data.shape)
        print("Number: ", number)
        fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return fps_data
    
    def __get_centroids(self, pcl):
        """
        Use farthest point sampling to return the centroids of the point cloud.
        NOTE: requires GPU
        """
        device = torch.device('cuda')
        pcl = torch.from_numpy(np.expand_dims(pcl, axis=0)).float().to(device)
        centroids = self.__fps(pcl, 64)
        centroids = centroids.detach().cpu().numpy().squeeze(0)
        return centroids

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
        dist[np.where(self.__ctr_dist(target) > self.__ctr_dist(current))] = 0

        # keep the N largest distances
        N = 200
        dist[np.argpartition(dist, N)[:N]] = 0

        point_a,point_b = np.random.choice(np.arange(idx.shape[0]),2,p=dist/np.sum(dist))
        pos_a = current[point_a,:]
        pos_b = current[point_b,:]
        vector = pos_a - pos_b
        distance = np.sqrt(np.sum(vector**2))
        rotation = self.__get_rotation_from_vector(vector)
        action = list((pos_a + pos_b)/2) + list(rotation) + [distance]
        # TODO: turn the distance into the distance betwen the two inward points (i.e. target[point_a, :], target[point_b, :])
        return action, pos_a, pos_b

        # need to take the x,y,z and divide by 10 add the center of the clay to it (point cloud is normalized and scaled???)
        # need to only take distances within the gripper's limits
        # return action
        # return idx, dist, point_a, point_b, pos_a, pos_b, vector, dist, rotation, action

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
        idx,dist = self.__match_point_pairs(current, target)
        point_a, point_b = idx[np.argpartition(dist,-2)[-2:]]
        pos_a = current[point_a,:]
        pos_b = current[point_b,:]
        vector = pos_a - pos_b
        vector += np.random.rand(3) * NOISY
        rotation = self.__get_rotation_from_vector(vector)
        distance = np.random.rand() * DISTANCE
        action = list((pos_a + pos_b)/2) + list(rotation) + [distance]
        return action, pos_a, pos_b
    
    def centroid_plan(self, current, target):
        '''
        '''
        self.__get_centroids(current)
        assert False
    
if __name__ == '__main__':
    args = None
    planner = GeometricSampler(args)

    # points_2 = np.random.random((2048,3))
    # # points_2 = np.random.random((2048,3))

    # # generate numpy array of random 3d points forming a sphere
    # points_1 = np.random.normal(size=(2048,3))

    path = '/home/alison/Clay_Data/Fully_Processed/May4_5D'
    points_1 = np.load(path + '/States/shell_scaled_state2010.npy')
    points_2 = np.load(path + '/Next_States/shell_scaled_next_state2015.npy')


    planner.centroid_plan(points_1, points_2)

    # # heavily downsampel the point cloud
    # downsample_scale = 4
    # points_1 = points_1[np.arange(0, points_1.shape[0], downsample_scale), :]
    # points_2 = points_2[np.arange(0, points_2.shape[0], downsample_scale), :]

    for i in range(10):
        # get the distances and some indication if the target is closer to the center than the current point
            # get the distances of each point to (0,0,0)
            # if ctr_dist(target) > ctr_dist(current):
            # dist[np.where(ctr_dist) > ctr_dist(current)] = 0
        # set all distances where the target is not closer to the center to be 0
        # 

        # TODO:
            # need to get a signed distance (i.e. only get positive distances (if the target point is closer to the center than the current point)
            # 

        action, pos_a, pos_b = planner.plan_a(points_1,points_2)

        # print("\nPlan a: ", planner.plan_a(points_1,points_2))
        # print("\nPlan b: ", planner.plan_b(points_1,points_2))

        # plot the two point clouds in different colors
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(points_1)
        pc1_colors = np.tile(np.array([0, 0, 1]), (len(points_1),1))
        pc1.colors = o3d.utility.Vector3dVector(pc1_colors)

        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(points_2)
        pc2_colors = np.tile(np.array([1, 0, 0]), (len(points_2),1))
        pc2.colors = o3d.utility.Vector3dVector(pc2_colors)

        # plot the plans in different colors
        # pos_a, pos_b, pos_a /pos_b +/- distance (along direction)
        plan_a_points = np.array([pos_a, pos_b])
        plan_a = o3d.geometry.PointCloud()
        plan_a.points = o3d.utility.Vector3dVector(plan_a_points)
        plan_a_colors = np.tile(np.array([0, 1, 0]), (len(plan_a_points),1))
        plan_a.colors = o3d.utility.Vector3dVector(plan_a_colors)
        o3d.visualization.draw_geometries([pc1, pc2, plan_a])





    # TODO: get the action in the clay frame (will need to unscale and unnormalize the action afterwards)


# def get_closest_point_pairs(current, target):
#     """
#     Given two point clouds, find the closest point pairs between them.
#     """
    
#     # ------- determine the closes point pairs -------
#     current_tree = KDTree(target)
#     dists, idxs = current_tree.query(current)
#     distances = dists.reshape(-1,1)
#     closest_pairs = np.column_stack((current, target[idxs]))
#     closest_pairs = np.concatenate((closest_pairs, distances), axis=1)
#     sorted_pairs = closest_pairs[(-distances).argsort()]
#     return sorted_pairs


# def geomatric_sampler(pc1, pc2):
#     """
#     Given two point clouds, find the 3D line passing through  (0, 0, 0)
#     where the difference between the points from the two point clouds on the line are greatest.
#     """

#     pass
