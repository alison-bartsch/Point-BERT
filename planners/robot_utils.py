import os
import cv2
import time
import copy
import math
import scipy
import random
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from skimage.color import rgb2lab
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def normalize_action(action, action_dim):
    if action_dim == 5:
        x = (action[0] - 0.55) / (0.63 - 0.55)
        y = (action[1] - (-0.035)) / (0.035 - (-0.035))
        # z = (action[2] - 0.178) / (0.2 - 0.178)
        z = (action[2] - 0.19) / (0.25 - 0.19)
        rz = (action[3] - (-90)) / (90 - (-90))
        d = (action[4] - 0.005) / (0.05 - 0.005)
        return np.array([x, y, z, rz, d])
    elif action_dim == 2:
        rz = (action[3] - (-90)) / (90 - (-90))
        d = (action[4] - 0.005) / (0.05 - 0.005)
        return np.array([rz, d])

def get_real_action_from_normalized(action_normalized):
    """
    """
    # value = (max - min)*(normalized) + min
    x = (0.63 - 0.55) * action_normalized[0] + 0.55
    y = (0.035 - (-0.035)) * action_normalized[1] + (-0.035)
    # z = (0.2 - 0.178) * action_normalized[2] + 0.178
    z = (0.25 - 0.19) * action_normalized[2] + 0.19
    rz = (90 - (-90)) * action_normalized[3] + (-90)
    d = (0.05 - 0.005) * action_normalized[4] + 0.005

    # clip values to avoid central mount
    if x >= 0.586 and x <= 0.608 and y >= -0.011 and y <=0.011 and z >= 0.173 and z <= 0.181:
        if x <= (0.586 + 0.608)/2.0:
            x = 0.584
        elif x > (0.586 + 0.608)/2.0:
            x = 0.61
        elif y <= 0:
            y = -0.013
        elif y > 0:
            y = 0.013
        elif z <= (0.173 + 0.181)/2.0:
            z = 0.17
        elif z > (0.173 + 0.181)/2.0:
            z = 0.184

    return np.array([x, y, z, rz, d])

def random_sample_action(action_dim):
    """
    Generate grasp action randomly. If action_dim == 5, then only rotation about z-axis,
    if action_dim == 7, then rotate about all axes.

    TODO: update for modified action space
    """
    x = round(random.uniform(0.55, 0.63), 3)
    y = round(random.uniform(-0.035, 0.035), 3)
    # z = round(random.uniform(0.178, 0.2), 3)
    z = round(random.uniform(0.19, 0.25), 3)
    d = round(random.uniform(0.005, 0.05), 3)
    rz = random.randint(-90, 90)

    # clip values to avoid central mount
    if x >= 0.586 and x <= 0.608 and y >= -0.011 and y <=0.011 and z >= 0.173 and z <= 0.181:
        if x <= (0.586 + 0.608)/2.0:
            x = 0.584
        elif x > (0.586 + 0.608)/2.0:
            x = 0.61
        elif y <= 0:
            y = -0.013
        elif y > 0:
            y = 0.013
        elif z <= (0.173 + 0.181)/2.0:
            z = 0.17
        elif z > (0.173 + 0.181)/2.0:
            z = 0.184

    if action_dim == 5:
        action = np.array([x, y, z, rz, d])
    elif action_dim == 7:
        rx = random.randint(-50, 50)
        ry = random.randint(-50, 50)
        action = np.array([x, y, z, rx, ry, rz, d])
    else:
        raise Exception("Action Dimension must be 5 or 7")
    
    return action

def random_sample_normalized_action(action_dim):
    """
    Generates normalized grasp action randomly.
    """
    if action_dim == 2:
        d = round(random.uniform(0,1), 3)
        rz = round(random.uniform(0,1), 3)
        return np.array([d, rz])
    elif action_dim == 5:
        x = round(random.uniform(0,1), 3)
        y = round(random.uniform(0,1), 3)
        z = round(random.uniform(0,1), 3)
        d = round(random.uniform(0,1), 3)
        rz = round(random.uniform(0,1), 3)
        return np.array([x, y, z, rz, d])

def random_sample_normalized_constrained_action(action_dim):
    """
    """
    if action_dim == 2:
        d = round(random.uniform(0,0.6), 3)
        rz = round(random.uniform(0,1), 3)
        return np.array([d, rz])
    elif action_dim == 5:
        x = round(random.uniform(0,1), 3)
        y = round(random.uniform(0,1), 3)
        z = round(random.uniform(0,1), 3)
        d = round(random.uniform(0,0.6), 3)
        rz = round(random.uniform(0,1), 3)
        return np.array([x, y, z, rz, d])
    
def plot_target_and_state_clouds(state, target):
    """
    """
    target_pcl = o3d.geometry.PointCloud()
    target_pcl.points = o3d.utility.Vector3dVector(target)
    target_colors = np.tile(np.array([0,0,1]), (len(target),1))
    target_pcl.colors = o3d.utility.Vector3dVector(target_colors)

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(state)
    pcl_colors = np.tile(np.array([1,0,0]), (len(state),1))
    pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    return target_pcl, pcl

def goto_grasp(fa, x, y, z, rx, ry, rz, d):
	"""
	Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
	This function was designed to be used for clay moulding, but in practice can be applied to any task.

	:param fa:  franka robot class instantiation
	"""
	pose = fa.get_pose()
	starting_rot = pose.rotation
	orig = Rotation.from_matrix(starting_rot)
	orig_euler = orig.as_euler('xyz', degrees=True)
	rot_vec = np.array([rx, ry, rz])
	new_euler = orig_euler + rot_vec
	r = Rotation.from_euler('xyz', new_euler, degrees=True)
	pose.rotation = r.as_matrix()
	pose.translation = np.array([x, y, z])

	fa.goto_pose(pose)
	fa.goto_gripper(d, force=60.0)
	time.sleep(3)
        
# def pcl_to_image(pcd1, pcd2, save_path):
#     """
#     """
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd1)
#     if pcd2 != False:
#         vis.add_geometry(pcd2)
#     ctr = vis.get_view_control()
#     ctr.change_field_of_view(step=-30.0)
#     vis.run()
#     image_top = vis.capture_screen_float_buffer(True)
#     plt.imsave(save_path, np.asarray(image_top))
#     vis.destroy_window()
#     del vis
#     del ctr
        
def pcl_to_image(pcd1, pcd2, save_path):
    """
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    if pcd2 != False:
        vis.add_geometry(pcd2)
    vis.run()
    image = vis.capture_screen_float_buffer(True)
    plt.imsave(save_path, np.asarray(image))
    vis.destroy_window()
    del vis

def emd(X, Y):
    """
    params: X   first point cloud (with batch size of 1)
    params: Y   second point cloud (with batch size of 1)
    """
    X = np.squeeze(X)
    Y = np.squeeze(Y)
    d = cdist(X, Y, 'euclidean')
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(X), len(Y))

	
def save_rgb_image(pipeline, aligned_stream, path):
    """
    """
    frames = pipeline.wait_for_frames()
    frames = aligned_stream.process(frames)
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(path, color_image)

def get_camera_point_cloud(pipeline, aligned_stream):
    """
    """
    W = 848
    H = 480

    frames = pipeline.wait_for_frames()
    frames = aligned_stream.process(frames)
    color_frame = frames.get_color_frame()
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    depth_frame = frames.get_depth_frame().as_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=intrinsics.fx, fy=intrinsics.fy, cx=intrinsics.ppx, cy=intrinsics.ppy)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    return pc

def remove_stage_grippers(pcd):

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    ind_z_upper = np.where(points[:,2] > 0.207)
    pcd.points = o3d.utility.Vector3dVector(points[ind_z_upper])
    pcd.colors = o3d.utility.Vector3dVector(colors[ind_z_upper])

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    ind_z_lower = np.where(points[:,2] < 0.27)
    pcd.points = o3d.utility.Vector3dVector(points[ind_z_lower])
    pcd.colors = o3d.utility.Vector3dVector(colors[ind_z_lower])

    # # PREVIOUS (BEFORE 8/29)
    # ind_z_upper = np.where(points[:,2] > 0.2325)
    # pcd.points = o3d.utility.Vector3dVector(points[ind_z_upper])
    # pcd.colors = o3d.utility.Vector3dVector(colors[ind_z_upper])

    # points = np.asarray(pcd.points)
    # colors = np.asarray(pcd.colors)
    # ind_z_lower = np.where(points[:,2] < 0.305)
    # pcd.points = o3d.utility.Vector3dVector(points[ind_z_lower])
    # pcd.colors = o3d.utility.Vector3dVector(colors[ind_z_lower])
    
    return pcd

def remove_background(pcd, radius=0.9, center = np.array([0, 0, 0])):
    """
    1. Accept raw point cloud or point cloud with outliers removed
    2. Crop points based on defined sphere parameters
    3. Return cropped point cloud
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    distances = np.linalg.norm(points - center, axis=1)
    indices = np.where(distances <= radius)

    pcd.points = o3d.utility.Vector3dVector(points[indices])
    pcd.colors = o3d.utility.Vector3dVector(colors[indices])
    
    return pcd

def lab_color_crop(pcd_incoming):
    """
    1. Creates copy of incoming pcd so as to not permanently alter original
    2. Defines vertices and color values of pcd copy
    3. Converts colors form RBG to LAB space
    4. Defines color threshold and applies threshold to verticies
    5. Returns thresholded pcd 
    """
    
    pcd = copy.deepcopy(pcd_incoming)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    lab_colors = rgb2lab(colors)

    a_idx = np.where(lab_colors[:,1]<-5)
    l_idx = np.where(lab_colors[:,0]>0)
    b_idx = np.where(lab_colors[:,2]>-5)

    indices = np.intersect1d(a_idx, np.intersect1d(l_idx, b_idx))

    pcd.points = o3d.utility.Vector3dVector(points[indices])
    pcd.colors = o3d.utility.Vector3dVector(colors[indices])
    
    return pcd

def fuse_point_clouds(pc2, pc3, pc4, pc5):
    """
    Updated fusal based on Charlotte's improvements
    """    
    # import the transforms
    transform_23 = np.load('planners/Camera_Extrinsics/transform_cam2_to_cam3_wonderful.npy')
    transform_34 = np.load('planners/Camera_Extrinsics/transform_cam3_to_cam4_perfect.npy')
    transform_54 = np.load('planners/Camera_Extrinsics/transform_cam5_to_cam4_perfect.npy')
    transform_4w = np.load('planners/Camera_Extrinsics/cam4_world_transform.npy')
    transform_w_improvement = np.load('planners/Camera_Extrinsics/world_transform.npy')

    # transform and combine all clouds
    pc2.transform(transform_23)
    pc2.transform(transform_34)
    pc2.transform(transform_4w)
    pc2.transform(transform_w_improvement)
    pc3.transform(transform_34)
    pc3.transform(transform_4w)
    pc3.transform(transform_w_improvement)
    pc4.transform(transform_4w)
    pc4.transform(transform_w_improvement)
    pc5.transform(transform_54)
    pc5.transform(transform_4w)
    pc5.transform(transform_w_improvement)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = pc5.points
    pointcloud.colors = pc5.colors
    pointcloud.points.extend(pc2.points)
    pointcloud.colors.extend(pc2.colors)
    pointcloud.points.extend(pc3.points)
    pointcloud.colors.extend(pc3.colors)
    pointcloud.points.extend(pc4.points)
    pointcloud.colors.extend(pc4.colors)

    o3d.visualization.draw_geometries([pointcloud])

    # ------ cropping point cloud ------ 
    pointcloud, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) # remove statistical outliers                       
    pointcloud = remove_stage_grippers(pointcloud)
    o3d.visualization.draw_geometries([pointcloud])
    pointcloud = remove_background(pointcloud, radius = .15, center = np.array([0.6, -0.05, 0.3]))
    o3d.visualization.draw_geometries([pointcloud])

    # ----- color thresholding -----
    pointcloud = lab_color_crop(pointcloud)
    o3d.visualization.draw_geometries([pointcloud])
    pointcloud, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # ----- calculate point cloud normals -----
    pointcloud.estimate_normals()

    # ------ downsample the point cloud -------
    downpdc = pointcloud.voxel_down_sample(voxel_size=0.0025)
    downpdc_points= np.asarray(downpdc.points)
    # print("min: ", np.amin(downpdc_points, axis=0))
    # print("max: ", np.amax(downpdc_points, axis=0))

    # ----- get shape of clay base --------
    # polygon_indices = np.where(downpdc_points[:,2] < 0.236) # PREVIOUS BEFORE 8/29
    polygon_indices = np.where(downpdc_points[:,2] < 0.22)

    # polygon_indices = np.where(downpdc_points[:,2] < 0.234)
    polygon_pcl = o3d.geometry.PointCloud()
    polygon_pcl.points = o3d.utility.Vector3dVector(downpdc_points[polygon_indices])

    # ------ generate a 2d grid of points for the base ---------
    base_plane = []
    minx, maxx = np.amin(downpdc_points[:,0]), np.amax(downpdc_points[:,0])
    miny, maxy = np.amin(downpdc_points[:,1]), np.amax(downpdc_points[:,1])
    minz, maxz = np.amin(downpdc_points[:,2]), np.amax(downpdc_points[:,2])
    x_vals = np.linspace(minx, maxx, 50)
    y_vals = np.linspace(miny, maxy, 50)
    xx, yy = np.meshgrid(x_vals, y_vals) # create grid that covers full area of 2d polygon  
    # z = 0.234 # height of the clay base

    z = 0.21
    # z = 0.232 # PREVIOUS BEFORE 8/29
    zz = np.ones(len(xx.flatten()))*z
    points = np.vstack((xx.flatten(), yy.flatten(), zz)).T

    grid_cloud = o3d.geometry.PointCloud()
    grid_cloud.points = o3d.utility.Vector3dVector(points)

    # -------- crop shape of clay base out of 2d grid of points ---------
    polygon_coords = np.asarray(polygon_pcl.points)[:,0:2]
    polygon = Polygon(polygon_coords)
    mask = [polygon.contains(Point(x, y)) for x, y in np.asarray(grid_cloud.points)[:,0:2]]
    cropped_grid = np.asarray(grid_cloud.points)[:,0:2][mask]
    zs = np.ones(len(cropped_grid))*z
    cropped_grid = np.concatenate((cropped_grid, np.expand_dims(zs, axis=1)), axis=1)

    base_cloud = o3d.geometry.PointCloud()
    base_cloud.points = o3d.utility.Vector3dVector(cropped_grid)

    # -------- add top part of clay to new base ------------
    base_cloud.points.extend(downpdc.points)
    cropped_plane, ind = base_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    base_cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(base_cloud.points),1)))
    o3d.visualization.draw_geometries([base_cloud])

    # uniformly sample 2048 points from each point cloud
    points = np.asarray(base_cloud.points)
    idxs = np.random.randint(0,len(points),size=2048)
    points = points[idxs]

    # re-process the processed_pcl to center
    pc_center = np.array([0.6, 0.0, 0.24])
    # pc_center = base_cloud.get_center() # TODO: have the center hard coded for consistency w.r.t. cloud
    centered_points = points - pc_center

    scale = 10
    rescaled_points = scale*centered_points

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(rescaled_points)
    pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(rescaled_points),1)))
    # o3d.visualization.draw_geometries([pointcloud])
    return rescaled_points

def wrap_rz(original_rz):
    """
    We want rz to be between -90 to 90, so wrap around if outside these bounds due to symmetrical gripper.
    """
    wrapped_rz = (original_rz + 90) % 180 - 90
    return wrapped_rz

def cloud_rotation_augmentation(np_cloud, save_path, start_save_idx, augment_action=False, action=None):
    # --------- augment data by rotating cloud and generating new grasp and save ---------
    rot_increment = 6 # [deg]
    COM = (0.6, 0.0)

    # convert pointcloud from numpy array to o3d point cloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np_cloud)

    if augment_action:
        og_global_grasp = (action[0], action[1])
        unit_circle_og_grasp = (og_global_grasp[0] - COM[0], og_global_grasp[1] - COM[1])
        unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
        rot_original = (math.degrees(math.acos(unit_circle_og_grasp[0] / unit_circle_radius)), math.degrees(math.asin(unit_circle_og_grasp[1] / unit_circle_radius))) # [deg]
        # NOTE: rot_original for x needs to be calculated with acos, and rot_original for accuraze y needs to be calculated with asin

    augmented_actions = []
    for i in range(59):
        full_save_path = save_path + '/shell_scaled_state' + str(start_save_idx + i+1) + '.npy'

        rot = (i+1)*rot_increment
        if augment_action:
            rot_new = (rot_original[0] + rot, rot_original[1] + rot)
            new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new[0])), unit_circle_radius*math.sin(math.radians(rot_new[1])))
            new_global_grasp = (COM[0] + new_unit_circle_grasp[0], COM[1] + new_unit_circle_grasp[1])

            # save new grasp to .npy file in augmented folder
            x = new_global_grasp[0]
            y = new_global_grasp[1]
            rz = action[5] + rot
            rz = wrap_rz(rz)
            grasp_action = np.array([x, y, action[2], action[3], action[4], rz, action[6]])
            augmented_actions.append(grasp_action)

        # apply the rotation to the point cloud (or voxelized cloud) and save
        pc_center = pointcloud.get_center()
        pcl = copy.deepcopy(pointcloud)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        R = mesh.get_rotation_matrix_from_xyz((0, 0, math.radians(rot)))
        pcl.rotate(R, center=pc_center)

        # print('\nRotated clay with base...\n')
        # o3d.visualization.draw_geometries([pcl])

        # uniformly sample 2048 points from each point cloud
        points = np.asarray(pcl.points)
        idxs = np.random.randint(0,len(points),size=2048)
        points = points[idxs]

        # re-process the processed_pcl to center
        pc_center = pointcloud.get_center()
        centered_points = points - pc_center
        scale = 10
        rescaled_points = scale*centered_points
        np.save(full_save_path, rescaled_points)
    return augmented_actions