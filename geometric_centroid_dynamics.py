import numpy as np
import open3d as o3d
from dynamics.dynamics_dataset import DemoActionDataset, GeometricDataset

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
    """
    Given the center point, rotation and length of the line, generate one in o3d format for plotting
    """
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

def line_3d_point_set(points):
    """
    Given a list of list of points, convert to a list of points and create fully connected lines.
    """
    # print("points: ", points)
    new_points = []
    for elem in points:
        # print("\nelem: ", elem)
        for i in range(2):
            # print("\narr: ", elem[i])
            new_points.append(elem[i])
    # print("New Points: ", new_points)

    lines = np.array([[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7],
                      [1,2], [1,3], [1,4], [1,5] , [1,6], [1,7],
                      [2,3], [2,4], [2,5], [2,6], [2,7],
                      [3,4], [3,5], [3,6], [3,7],
                      [4,5], [4,6], [4,7],
                      [5,6], [5,7],
                      [6,7]])
    return new_points, lines

def points_inside_rectangle(rectangle_points, push_dir, pcl):
    """
    Given a 3D rectangular shape and a point cloud, return the points inside the rectangle, and the
    distance to move (in the direction of the push) to be outside of the rectangle.
    """
    # return points in the format [[x1,y1,z1], [x2,y2,z2], ...] - this is for o3d line set visualization
    # return associated lines indicating which points we want lines drawn  between i.e. [[0,1], [0,2], ...]

    # TODO: THIS FUNCTION NOT WORKING CORRECTLY RIGHT NOW

    # print("\nRectangle Points: ", rectangle_points)
    # print("Min: ", np.min(rectangle_points, axis=0))

    # assert False

    # minx, miny, minz = np.min(rectangle_points, axis=0)
    # maxx, maxy, maxz = np.max(rectangle_points, axis=0)

    # print("minx: ", minx)
    # print("miny: ", miny)
    # print("minz: ", minz)
    # print("maxx: ", maxx)
    # print("maxy: ", maxy)
    # print("maxz: ", maxz)

    # inside_indices = np.where((pcl[:,0] > minx) & (pcl[:,0] < maxx) & (pcl[:,1] > miny) & (pcl[:,1] < maxy) & (pcl[:,2] > minz) & (pcl[:,2] < maxz))[0]

    # print("og pcl: ", pcl.shape)
    # print("inside: ", pcl[inside_indices,:].shape)
    # return inside_indices

    # assert False

    min_corner = np.min(rectangle_points, axis=0)
    max_corner = np.max(rectangle_points, axis=0)
    # print("Min corner: ", min_corner)
    # print("Max corner: ", max_corner)

    inside_mask = np.all((min_corner <= pcl) & (pcl <= max_corner), axis=1)
    inside_indices = np.where(inside_mask)[0]
    return inside_indices

def dir_vec_from_points(pt1, pt2):
    """
    Given two 3D points, find the direction vector from pt1 to pt2 and return
    the unit vector.
    """
    dir_vec = pt2 - pt1
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    return dir_vec

# path
# path = '/home/alison/Clay_Data/Fully_Processed/Aug15_5D_Human_Demos'
# path = '/home/alison/Clay_Data/Fully_Processed/Aug24_Human_Demos_Fully_Processed'
path = '/home/alison/Clay_Data/Fully_Processed/Aug29_Human_Demos'
# initialize the dataset
dataset = GeometricDataset(path, 'shell_scaled')
# import an action and a state
# state, next_state, action = dataset.__getitem__(0)


test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120, 6000]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)

    state = state.detach().numpy()
    action = action.detach().numpy()

    # unnormalize the action
    action = unnormalize_a(action)

    # center the action at the origin of the point cloud
    # pcl_center = np.array([0.6, 0.0, 0.24]) # verified same pcl center that processed point clouds
    pcl_center = np.array([0.6, 0.0, 0.25])
    action[0:3] = action[0:3] - pcl_center
    action[0:3] = action[0:3] + np.array([0.005, -0.002, 0.0]) # observational correction
    # print("action centered: ", action)

    # scale the action (multiply x,y,z,d by 10)
    action_scaled = action * 10
    action_scaled[3] = action[3] # don't scale the rotation
    # print("Action Scaled:", action_scaled)
    # plot an 8cm (x10 to scale) line centered at x,y,z with rotation rx, ry, rz [currently just rz]
    len = 10 * 0.08 # 8cm scaled  to point cloud scaling # TODO: figure out grasp width scaling issue
    # len = 1.6

    # NOTE: issue with the distance between fingertips in the unnormalized action
        # appears to be outside of the acceptable range from 0.05 to 0.005 meters
        
    # get the points and lines for the action orientation visualization
    ctr = action_scaled[0:3]
    upper_ctr = ctr + np.array([0,0, 0.6])
    # rz = 90 + action_scaled[3] # NOTE: This is in degrees
    rz = 90 + action_scaled[3]
    # rz = 75 + action_scaled[3] # TODO: add the original rotation instead of 90 (it's slightly off)
    points, lines = line_3d_start_end(ctr, rz, len)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (lines.shape[0],1)))

    # greate point cloud for the gripper action in red
        # (1) get the coordinates of a rectangle with the bottom centered at the end of that line
        # (2) gripper width originally 1.8 cm
        # (3) gripper height originally 6 cm

    # get the end points for the grasp
    # delta = 2*0.5*(0.8 - action_scaled[4]) # NOTE: seems to be 2x scaled, but shouldn't be!

    delta = 0.8 - action_scaled[4] 
    end_pts, _ = line_3d_start_end(ctr, rz, len - delta)
    top_end_pts, _ = line_3d_start_end(upper_ctr, rz, len - delta)

    # get the top points for the grasp (given gripper finger height)
    top_points, _ = line_3d_start_end(upper_ctr, rz, len)

    # gripper 1 
    g1_base_start, _ = line_3d_start_end(points[0], rz+90, 0.18)
    g1_base_end, _ = line_3d_start_end(end_pts[0], rz+90, 0.18)
    g1_top_start, _ = line_3d_start_end(top_points[0], rz+90, 0.18)
    g1_top_end, _ = line_3d_start_end(top_end_pts[0], rz+90, 0.18)
    g1_points, g1_lines = line_3d_point_set([g1_base_start, g1_base_end, g1_top_start, g1_top_end])

    # get the points in state pcl inside gripper 1
    g1_idx = points_inside_rectangle(g1_points, None, state)
    inlier_pts = state.copy()

    # pointcloud with points inside rectangle
    g1_inside = state[g1_idx,:]
    g1_inside_pcl = o3d.geometry.PointCloud()
    g1_inside_pcl.points = o3d.utility.Vector3dVector(g1_inside)
    g1_inside_colors = np.tile(np.array([1, 0, 0]), (g1_inside.shape[0],1))
    g1_inside_pcl.colors = o3d.utility.Vector3dVector(g1_inside_colors)

    # get the displacement vector for the gripper 1 base
    g1_dir_unit = dir_vec_from_points(end_pts[0], points[0])
    g1_displacement_vec = end_pts[0] - points[0]

    # apply the displacement vector to all the points in the state point cloud
    g1_diffs = np.tile(end_pts[0], (inlier_pts[g1_idx,:].shape[0],1)) - inlier_pts[g1_idx,:] 
    g1_diffs = np.linalg.norm(g1_diffs, axis=1)
    # inlier_pts[g1_idx,:] = inlier_pts[g1_idx,:] + np.tile(g1_displacement_vec, (inlier_pts[g1_idx,:].shape[0],1))
    inlier_pts[g1_idx,:] = inlier_pts[g1_idx,:] -  np.tile(g1_diffs, (3,1)).T * np.tile(g1_dir_unit, (inlier_pts[g1_idx,:].shape[0],1))

    g1 = o3d.geometry.LineSet()
    g1.points = o3d.utility.Vector3dVector(g1_points)
    g1.lines = o3d.utility.Vector2iVector(g1_lines)
    g1.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g1_lines.shape[0],1)))

    # gripper 2
    g2_base_start, _ = line_3d_start_end(points[1], rz+90, 0.18)
    g2_base_end, _ = line_3d_start_end(end_pts[1], rz+90, 0.18)
    g2_top_start, _ = line_3d_start_end(top_points[1], rz+90, 0.18)
    g2_top_end, _ = line_3d_start_end(top_end_pts[1], rz+90, 0.18)
    g2_points, g2_lines = line_3d_point_set([g2_base_start, g2_base_end, g2_top_start, g2_top_end])

    # get the points in state pcl inside gripper 1
    g2_idx = points_inside_rectangle(g2_points, None, state)

    # pointcloud with points inside rectangle
    g2_inside = state[g2_idx,:]
    g2_inside_pcl = o3d.geometry.PointCloud()
    g2_inside_pcl.points = o3d.utility.Vector3dVector(g2_inside)
    g2_inside_colors = np.tile(np.array([1, 0, 0]), (g2_inside.shape[0],1))
    g2_inside_pcl.colors = o3d.utility.Vector3dVector(g2_inside_colors)

    # get the displacement vector for the gripper 1 base
    g2_dir_unit = dir_vec_from_points(end_pts[1], points[1])
    g2_displacement_vec = end_pts[1] - points[1]

    # apply the displacement vector to all the points in the state point cloud
    g2_diffs = np.tile(end_pts[1], (inlier_pts[g2_idx,:].shape[0],1)) - inlier_pts[g2_idx,:] 
    g2_diffs = np.linalg.norm(g2_diffs, axis=1)
    # inlier_pts[g1_idx,:] = inlier_pts[g1_idx,:] + np.tile(g1_displacement_vec, (inlier_pts[g1_idx,:].shape[0],1))
    
    inlier_pts[g2_idx,:] = inlier_pts[g2_idx,:] -  np.tile(g2_diffs, (3,1)).T * np.tile(g2_dir_unit, (inlier_pts[g2_idx,:].shape[0],1))
    inliers = o3d.geometry.PointCloud()
    inliers.points = o3d.utility.Vector3dVector(inlier_pts)
    inlier_colors = np.tile(np.array([1, 0, 0]), (inlier_pts.shape[0],1))
    inliers.colors = o3d.utility.Vector3dVector(inlier_colors)

    g2 = o3d.geometry.LineSet()
    g2.points = o3d.utility.Vector3dVector(g2_points)
    g2.lines = o3d.utility.Vector2iVector(g2_lines)
    g2.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g2_lines.shape[0],1)))

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
    o3d.visualization.draw_geometries([og_pcl, ns_pcl, ctr_action, line_set, g1, g2])
    o3d.visualization.draw_geometries([og_pcl, ns_pcl, ctr_action, line_set, g1, g2, g1_inside_pcl, g2_inside_pcl])
    o3d.visualization.draw_geometries([ns_pcl, ctr_action, line_set, g1, g2, inliers])
    # o3d.visualization.draw_geometries([og_pcl, ns_pcl, ctr_action, line_set, g1_start, g1_end, 
    #                                    g1_top_start, g2_start, g2_end])


# plot the dots at either end of the line as gripper starting
# given scaled d, plot the dot at the distance (8-d)*0.5 away from either end as the gripper ending

# given the gripper dimensions, plot a rectangular surface representing the grasp movements


# print out the points in the point cloud that are within the grasp surface
# move each of these points in the normal direction to the edge of the grasp surface