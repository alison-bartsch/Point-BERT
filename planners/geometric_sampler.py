import torch
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
# import planner_utils as u
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

    # def __calculate_dist(self, current, target):
    #     '''
    #     This function is used to calculate the distance between each point
    #     input: 
    #         current -- the 3d position of the point in point cloud, shape is [num_points, 3]
    #         target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    #     output: 
    #         dist  -- the matrix indicating the distance, the shape is [num_points, num_points] 
    #     '''

    #     assert current.shape[0] == target.shape[0] # make sure the number of points is the same
    #     dist = cdist(current,target,metric= 'euclidean')
    #     return dist
        
    # def __match_point_pairs(self, current, target):
    #     '''
    #     This function is used to match the points with their nearest neighbor
    #     input:
    #         current -- the 3d position of the point in point cloud, shape is [num_points, 3]
    #         target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    #     output:
    #         indice -- the array of the nearest neighbor's index for current points
    #         distance -- the array of the nearest neighbor's distance for current points
    #     ''' 
    #     dist = self.__calculate_dist(current, target)
    #     indice = np.argmin(dist,axis=1)
    #     distance = np.min(dist,axis=1)
    #     return indice,distance
    
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
    
    # def __ctr_dist(self, pcl):
    #     """
    #     Given a point cloud, get the distance from (0,0,0) to all of the points in the cloud
    #     """

    #     center = np.zeros((1,3)) # np.zeros((pcl.shape[0],3))
    #     dist = cdist(center,pcl,metric= 'euclidean')
    #     dist = dist.reshape(-1)
    #     return dist
    
    # # def __get_unique_matching_points(self, current, target):
    # #     '''
    # #     This function is used to get the corresponding points in target point cloud for each point in current point cloud
    # #     '''
    # #     tree = KDTree(target)
    # #     matches = []
    # #     dists = []

    # #     used_indices = set()

    # #     for point in current:
    # #         dist, idx = tree.query(point)
    # #         if dist 
    # #     return idx, dist
    
    # def __fps(self, data, number):
    #     '''
    #         data B N 3
    #         number int
    #     '''
    #     print("Data: ", data.shape)
    #     print("Number: ", number)
    #     fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    #     fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    #     return fps_data
    
    # def __get_centroids(self, pcl):
    #     """
    #     Use farthest point sampling to return the centroids of the point cloud.
    #     NOTE: requires GPU
    #     """
    #     device = torch.device('cuda')
    #     pcl = torch.from_numpy(np.expand_dims(pcl, axis=0)).float().to(device)
    #     centroids = self.__fps(pcl, 64)
    #     centroids = centroids.detach().cpu().numpy().squeeze(0)
    #     return centroids

    # def plan_a(self, current, target):
    #     '''
    #     Here is the implement of Strategy A
    #         1. Match the point with its neareat point;
    #         2. The distance between them is the sample weight;
    #         3. Random sample two points, take the center of them as the picking position, and the distance for the gripper closing.
    #     input:
    #         current -- the 3d position of the point in point cloud, shape is [num_points, 3]
    #         target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    #     output:
    #         action -- list, the action that random sampled, the dimension is [x,y,z,roll,pitch,yaw,d]
    #     '''
    #     idx,dist = self.__match_point_pairs(current, target)
    #     dist[np.where(self.__ctr_dist(target) > self.__ctr_dist(current))] = 0

    #     # keep the N largest distances
    #     N = 200
    #     dist[np.argpartition(dist, N)[:N]] = 0

    #     point_a,point_b = np.random.choice(np.arange(idx.shape[0]),2,p=dist/np.sum(dist))
    #     pos_a = current[point_a,:]
    #     pos_b = current[point_b,:]

    #     vector = pos_a - pos_b
    #     distance = np.sqrt(np.sum(vector**2))
    #     rotation = self.__get_rotation_from_vector(vector)
    #     action = list((pos_a + pos_b)/2) + list(rotation) + [distance]
    #     # TODO: turn the distance into the distance betwen the two inward points (i.e. target[point_a, :], target[point_b, :])
    #     return action, pos_a, pos_b

    #     # need to take the x,y,z and divide by 10 add the center of the clay to it (point cloud is normalized and scaled???)
    #     # need to only take distances within the gripper's limits
    #     # return action
    #     # return idx, dist, point_a, point_b, pos_a, pos_b, vector, dist, rotation, action

    # def plan_b(self, current, target):
    #     '''
    #     Here is the implement of Strategy B
    #         1. Match the point with its neareat point;
    #         2. Take the top two points with largest distance;
    #         3. Get the vector of these two points; 
    #         4. Around this vector, randomly sample actions.
    #     input:
    #         current -- the 3d position of the point in point cloud, shape is [num_points, 3]
    #         target -- the 3d position of the point in target point cloud, shape is [num_points, 3]
    #     output:
    #         action -- list, the action that random sampled, the dimension is [x,y,z,roll,pitch,yaw,d]
    #     '''
    #     idx,dist = self.__match_point_pairs(current, target)
    #     point_a, point_b = idx[np.argpartition(dist,-2)[-2:]]
    #     pos_a = current[point_a,:]
    #     pos_b = current[point_b,:]
    #     vector = pos_a - pos_b
    #     vector += np.random.rand(3) * NOISY
    #     rotation = self.__get_rotation_from_vector(vector)
    #     distance = np.random.rand() * DISTANCE
    #     action = list((pos_a + pos_b)/2) + list(rotation) + [distance]
    #     return action, pos_a, pos_b
    
    # def centroid_plan(self, current, target):
    #     '''
    #     Implementation of region-based geometric sampling strategy.
    #     '''
    #     # get the centroids of the point clouds
    #     current_centers = self.__get_centroids(current)
    #     target_centers = self.__get_centroids(target)

    #     # get the associated neighbor of each of the centers in the other cloud
    #     # want match point pairs to return a list of point pairs i.e. [(ctr1, ctr2), (ctr3, ctr4), ...)]
    #     idx, dist = self.__match_point_pairs(current_centers, target_centers)
    #     # print("\nMatching idxs: ", idx.shape)
    #     # print("Dist shape: ", dist.shape)
    #     # TODO: explore doing an exclusive match (i.e. each point can only be matched to one other point)
        
    #     # get the distances and set the sampling probability propertional to the distance (i.e. larger distance = more likely)
    #     sample_prob = dist/np.sum(dist)
    #     # print("\nSample Prob: ", sample_prob)
        
    #     # sample a pair of centers
    #     sample_idx = np.random.choice(np.arange(idx.shape[0]),1,p=dist/np.sum(dist))
        
    #     # get the vector towards the target centroid
    #     dir_vector = target_centers[idx[sample_idx], :][0] - current_centers[sample_idx, :][0]
    #     dir_vector /= np.linalg.norm(dir_vector)
    #     print("Dir Vector: ", dir_vector)
        
    #     # set action center to the target center
    #     action_pos = target_centers[idx[sample_idx], :][0] / 10.0 # rescale
    #     # print("Action: ", action_pos)
        
    #     # the action rotation should be aligned with the direction vector
    #     yaw = np.degrees(np.arctan2(dir_vector[1], dir_vector[0]))
    #     pitch = np.degrees(np.arcsin(-dir_vector[2]))
    #     roll = np.degrees(np.arctan2(-dir_vector[1], -dir_vector[2]))
    #     rotation = np.array([roll, pitch, yaw])
    #     print("Rotation: ", rotation)
        
    #     # the distance to grasp is the distance between the two centroids
    #     distance = dist[sample_idx][0] / 10.0 # rescale
    #     print("Distance: ", distance)

    #     # TODO: scale the position and rotation back to the original scale (i.e. divide by 10 and add the center of the clay)
        
    #     # return action        
    #     action = np.array([action_pos[0], action_pos[1], action_pos[2], rotation[0], rotation[1], rotation[2], distance])
    #     print("Action: ", action)
    #     return action, current_centers[sample_idx, :][0]
    
def divide_point_cloud(point_cloud, num_regions):
    kmeans = KMeans(n_clusters=num_regions, random_state=0)
    labels = kmeans.fit_predict(point_cloud)
    return labels

def region_distance(region1, region2, distances):
    distances_subset = distances[np.ix_(region1, region2)]
    return np.max(distances_subset)

def most_different_regions(pc1, pc2, num_regions):
    dist1_to_2 = cdist(pc1, pc2, metric='euclidean')
    labels1 = divide_point_cloud(pc1, num_regions)
    labels2 = divide_point_cloud(pc2, num_regions)

    most_different = []
    for idx1 in range(num_regions):
        for idx2 in range(num_regions):
            region1_idx = np.where(labels1 == idx1)[0]
            region2_idx = np.where(labels2 == idx2)[0]

            distance = region_distance(region1_idx, region2_idx, dist1_to_2)
            most_different.append((idx1, idx2, distance))
    
    # order from smallest to largest distance
    most_different.sort(key=lambda x: x[2], reverse=False)
    
    pcl1_clusters = []

    cluster_pairs = []
    for region in most_different:
        pt1 = region[0]
        pt2 = region[1]
        dist = region[2]
        
        if pt1 not in pcl1_clusters:
            pcl1_clusters.append(pt1)
            cluster_pairs.append((pt1, pt2, dist))
    
    cluster_pairs.sort(key=lambda x: x[2], reverse=True)
    print("Cluster Pairs: ", cluster_pairs)
    # most_different.sort(key=lambda x: x[2], reverse=True)

    # print("Most different: ", most_different)
    # print(len(most_different))

    # TODO: go from cluster pairs to an action!!!!
    return cluster_pairs, labels1, labels2
    
if __name__ == '__main__':
    args = None
    planner = GeometricSampler(args)

    # points_2 = np.random.random((2048,3))
    # # points_2 = np.random.random((2048,3))

    # # generate numpy array of random 3d points forming a sphere
    # points_1 = np.random.normal(size=(2048,3))

    path = '/home/alison/Clay_Data/Fully_Processed/May4_5D'
    points_1 = np.load(path + '/States/shell_scaled_state2030.npy')
    points_2 = np.load(path + '/Next_States/shell_scaled_next_state2035.npy')

    # plot the two point clouds in different colors
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(points_1)
    pc1_colors = np.tile(np.array([0, 0, 1]), (len(points_1),1))
    pc1.colors = o3d.utility.Vector3dVector(pc1_colors)

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(points_2)
    pc2_colors = np.tile(np.array([1, 0, 0]), (len(points_2),1))
    pc2.colors = o3d.utility.Vector3dVector(pc2_colors)

    cluster_pairs, labels1, labels2 = most_different_regions(points_1, points_2, 45) # original: 64
    print(len(cluster_pairs))

    for i in range(10):
        region_pair = cluster_pairs[i]
        point1 = region_pair[0]
        point2 = region_pair[1]
        dist = region_pair[2]

        # print("Point1: ", point1)
        # print("Point2: ", point2)
        # print("Dist: ", dist)

        region1_idx = np.where(point1 == labels1)[0]
        region1 = points_1[region1_idx, :]
        
        region2_idx = np.where(point2 == labels2)[0]
        region2 = points_2[region2_idx, :]

        # convert to green points and visualize
        reg1 = o3d.geometry.PointCloud()
        reg1.points = o3d.utility.Vector3dVector(region1)
        reg1_colors = np.tile(np.array([0, 1, 0]), (len(region1),1))
        reg1.colors = o3d.utility.Vector3dVector(reg1_colors)

        reg2 = o3d.geometry.PointCloud()
        reg2.points = o3d.utility.Vector3dVector(region2)
        reg2_colors = np.tile(np.array([0, 1, 1]), (len(region2),1))
        reg2.colors = o3d.utility.Vector3dVector(reg2_colors)
        o3d.visualization.draw_geometries([pc1, pc2, reg1, reg2])

        
    assert False

    for i in range(10):
        action, starting_point = planner.centroid_plan(points_1, points_2)

        # plot the two point clouds in different colors
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(points_1)
        pc1_colors = np.tile(np.array([0, 0, 1]), (len(points_1),1))
        pc1.colors = o3d.utility.Vector3dVector(pc1_colors)

        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(points_2)
        pc2_colors = np.tile(np.array([1, 0, 0]), (len(points_2),1))
        pc2.colors = o3d.utility.Vector3dVector(pc2_colors)

        # plot the gripper starting and end point
        pts = np.array([starting_point, action[:3]])
        pts = pts.reshape(-1,3)
        print("Points: ", pts.shape)
        pc3 = o3d.geometry.PointCloud()
        pc3.points = o3d.utility.Vector3dVector(pts)
        pc3_colors = np.tile(np.array([0, 1, 0]), (len(pts),1))
        pc3.colors = o3d.utility.Vector3dVector(pc3_colors)

        # create mesh arrow
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.07, cone_height=0.02)
        arrow.paint_uniform_color([0, 0, 0])
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(np.array([action[3], action[4], action[5]]))
        arrow.transform(rot_mat)

        # create coordinate frame at center of point cloud
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pc1, pc2, pc3, frame, arrow])

    assert False

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
