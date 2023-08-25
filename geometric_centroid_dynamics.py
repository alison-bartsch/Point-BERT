import numpy as np
import open3d as o3d
from dynamics.dynamics_dataset import DemoActionDataset

def normalize_a(action):
    if action.shape[0] == 2:
        a_mins2d = np.array([-90, 0.005])
        a_maxs2d = np.array([90, 0.05])
        norm_action = (action - a_mins2d) / (a_maxs2d - a_mins2d)

    elif action.shape[0] == 5:
        a_mins5d = np.array([0.55, -0.035, 0.19, -90, 0.005])
        a_maxs5d = np.array([0.63, 0.035, 0.25, 90, 0.05])
        norm_action = (action - a_mins5d) / (a_maxs5d - a_mins5d)
    
    elif action.shape[0] == 7:
        a_mins = np.array([0.55, -0.035, 0.19, -50, -50, -90, 0.005])
        a_maxs = np.array([0.63, 0.035, 0.25, 50, 50, 90, 0.05])
        norm_action = (action - a_mins) / (a_maxs - a_mins)
    
    else:
        print("\nERROR: action dimension incorrect\n")
        assert False
    
    return norm_action

def unnormalize_a(norm_action):
    if norm_action.shape[0] == 2:
        a_mins2d = np.array([-90, 0.005])
        a_maxs2d = np.array([90, 0.05])
        action = norm_action * (a_maxs2d - a_mins2d) + a_mins2d

    elif norm_action.shape[0] == 5:
        a_mins5d = np.array([0.55, -0.035, 0.19, -90, 0.005])
        a_maxs5d = np.array([0.63, 0.035, 0.25, 90, 0.05])
        action = norm_action * (a_maxs5d - a_mins5d) + a_mins5d
    
    elif norm_action.shape[0] == 7:
        a_mins = np.array([0.55, -0.035, 0.19, -50, -50, -90, 0.005])
        a_maxs = np.array([0.63, 0.035, 0.25, 50, 50, 90, 0.05])
        action = norm_action * (a_maxs - a_mins) + a_mins
    
    else:
        print("\nERROR: action dimension incorrect\n")
        assert False
    
    return action

def line_3d_start_end(center, rz, length):
    # convert rz to radians
    rz = np.radians(rz)
    dir_vec = np.array([np.cos(rz), np.sin(rz), 0])
    displacement = dir_vec * (0.5*length)
    start_point = center - displacement
    end_point = center + displacement
    points = np.array([start_point, end_point])
    # print("points: ", points)
    lines = np.array([[0,1]])
    return points, lines

    # return points in the format [[x1,y1,z1], [x2,y2,z2], ...] - this is for o3d line set visualization
    # return associated lines indicating which points we want lines drawn  between i.e. [[0,1], [0,2], ...]

# path
path = '/home/alison/Clay_Data/Fully_Processed/Aug15_5D_Human_Demos'
# initialize the dataset
dataset = DemoActionDataset(path, 'shell_scaled')
# import an action and a state
# state, next_state, action = dataset.__getitem__(0)


test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120, 6000, 7048]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)

    state = state.detach().numpy()
    action = action.detach().numpy()
    print("\n\nNormalized Action: ", action)
    print("Max State: ", np.amax(state, axis=0))
    print("Min state: ", np.amin(state, axis=0))
    # unnormalize the action
    action = unnormalize_a(action)
    print("Unnormalized action: ", action)
    # center the action at the origin of the point cloud
    pcl_center = np.array([0.59, 0.0, 0.22])
    action[0:3] = action[0:3] - pcl_center
    print("action centered: ", action)
    # scale the action (multiply x,y,z,d by 10)
    action_scaled = action * 10
    action_scaled[3] = action[3] # don't scale the rotation
    print("Action Scaled:", action_scaled)
    # plot an 8cm (x10 to scale) line centered at x,y,z with rotation rx, ry, rz [currently just rz]
    # len = 10 * 0.08 # 8cm scaled  to point cloud scaling # TODO: figure out grasp width scaling issue
    len = 2.2

    # NOTE: issue with the distance between fingertips in the unnormalized action
        # appears to be outside of the acceptable range from 0.05 to 0.005 meters
        

    ctr = action_scaled[0:3]
    rz = 90 + action_scaled[3] # NOTE: This is in degrees
    # rz = action_scaled[3] - 90
    points, lines = line_3d_start_end(ctr, rz, len)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (lines.shape[0],1)))

    # test plot the point cloud and action and next state
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(state)
    og_colors = np.tile(np.array([0, 0, 1]), (state.shape[0],1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

    ns_pcl = o3d.geometry.PointCloud()
    ns = next_state.detach().numpy()
    ns_pcl.points = o3d.utility.Vector3dVector(ns)
    ns_colors = np.tile(np.array([0, 1, 0]), (ns.shape[0],1))
    ns_pcl.colors = o3d.utility.Vector3dVector(ns_colors)

    ctr_action = o3d.geometry.PointCloud()
    action_cloud = action_scaled[0:3].reshape(1,3)
    print("Action: ", action_cloud)
    ctr_action.points = o3d.utility.Vector3dVector(action_scaled[0:3].reshape(1,3))
    ctr_colors = np.tile(np.array([1, 0, 0]), (1,1))
    ctr_action.colors = o3d.utility.Vector3dVector(ctr_colors)
    o3d.visualization.draw_geometries([og_pcl, ns_pcl, ctr_action, line_set])


# plot the dots at either end of the line as gripper starting
# given scaled d, plot the dot at the distance (8-d)*0.5 away from either end as the gripper ending

# given the gripper dimensions, plot a rectangular surface representing the grasp movements


# print out the points in the point cloud that are within the grasp surface
# move each of these points in the normal direction to the edge of the grasp surface