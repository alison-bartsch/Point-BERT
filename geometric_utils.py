import torch.nn as nn
import torch
import numpy as np
import networkx as nx
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from pointnet2_ops import pointnet2_utils

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
    min_corner = np.min(rectangle_points, axis=0)
    max_corner = np.max(rectangle_points, axis=0)
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

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    data = data.to(dtype=torch.float32)
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    if src.dtype != dst.dtype:
        dst = dst.double()
        src = src.double()

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center
    
    def get_neighborhood(self, xyz, center):
        """
        Given the centers, return the neighborhood
        """
        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

def create_graph_from_point_cloud(pcl, k=8):
    """
    Create a graph from a point cloud, assuming the point cloud is a numpy array.
    The edges are created by connecting each point to its k nearest neighbors.
    """
    
    # create KDTree to quickly calculate nearest neighbors
    kdtree = KDTree(pcl)
    # initialize undirected graph
    g = nx.Graph()
    # add nodes to the graph
    for idx, point in enumerate(pcl):
        g.add_node(idx, pos=point)
    # add edges to the graph based on the k nearest neighbors
    for idx, point in enumerate(pcl):
        point = np.expand_dims(point, axis=0)
        # get the k nearest neighbors
        _, neighbors = kdtree.query(point, k=k)
        # add edges to the graph
        for neighbor in neighbors[0][1:]:
            g.add_edge(idx, neighbor)
    return  g

def create_point_cloud_from_graph(G):
    """
    Create a point cloud from a graph, assuming the graph is a networkx graph.
    """
    node_positions = nx.get_node_attributes(G, 'pos')
    pcl = np.array(list(node_positions.values()))
    return pcl

def force_directed_layout(graph, displacements, iterations=100, k=0.1, spring_length=1.0):
    """
    Given a graph and a set of displacements, perform force-directed layout on the graph.
    """
    positions = nx.get_node_attributes(graph, 'pos')
    for _ in range(iterations):
        new_pos = {}
        for node in graph.nodes:

            displacement = np.array([0,0,0])
            
            # calculate the repulsion force from all nodes
            for other_node in graph.nodes:
                if node != other_node:
                    diff = positions[node] - positions[other_node]
                    displacement += k / np.linalg.norm(diff) ** 2 * diff

            # calculate the attraction force from connected nodes
            for neighbor in graph.neighbors(node):
                diff = positions[neighbor] - positions[node]
                displacement += spring_length ** 2 / k * diff
            
            new_pos[node] = positions[node] + displacements[node]
    return new_pos

def propagate_displacement(graph, node_to_move, displacement, max_edge_length=0.15, min_edge_length=0.005, max_iterations=100):
    """
    Given a graph and a single displacement, propagate the displacement through the graph maintaining
    edge length constraints.
    """
    # create a copy of the original graph to work with
    updated_graph = graph.copy()
    # move the target node by the specified displacement
    updated_graph.nodes[node_to_move]['pos'] += displacement

    # print("Test: ", updated_graph.nodes[0,1,2,3]['pos'])

    
    # node_ids = [list(updated_graph.nodes)[idx] for idx in nodes_to_move]
    # G_nodes = np.array(list(updated_graph.nodes))
    # print(G_nodes[nodes_to_move])
    # G_nodes[nodes_to_move] += displacements
    # np_pos = np.array([updated_graph.nodes[node_id]['pos'] for node_id in G_nodes])

    # for idx, node_id in enumerate(node_ids):
    #     updated_graph.nodes[node_id]['pos'] = np_pos[idx]
    

    for _ in range(max_iterations):
        for node in updated_graph.nodes:
            if node != node_to_move:
                # get the neighbors of the current node
                neighbors = list(updated_graph.neighbors(node))
                # adjust the node positions to satisfy edge length constraints
                for neighbor in neighbors:
                    current_length = np.linalg.norm(np.array(updated_graph.nodes[node]['pos']) - np.array(updated_graph.nodes[neighbor]['pos']))
                    if current_length > max_edge_length:
                        dir = np.array(updated_graph.nodes[neighbor]['pos']) - np.array(updated_graph.nodes[node]['pos'])
                        dir /= np.linalg.norm(dir)
                        disp = (current_length - max_edge_length) * dir * 0.5

                        # update positions
                        updated_graph.nodes[node]['pos'] += disp
                        updated_graph.nodes[neighbor]['pos'] -= disp

                    elif current_length < min_edge_length:
                        dir = np.array(updated_graph.nodes[neighbor]['pos']) - np.array(updated_graph.nodes[node]['pos'])
                        dir /= np.linalg.norm(dir)
                        disp = (min_edge_length - current_length) * dir * 0.5

                        # update positions
                        updated_graph.nodes[node]['pos'] -= disp
                        updated_graph.nodes[neighbor]['pos'] += disp

        # check if the edge length constraints are satisfied
        edge_lengths = [(u, v, np.linalg.norm(np.array(updated_graph.nodes[u]['pos']) - np.array(updated_graph.nodes[v]['pos']))) for u, v in updated_graph.edges]

        if all(length <= max_edge_length for _, _, length in edge_lengths):
            break

    return updated_graph

def apply_displacements_to_nodes(graph, list_of_nodes, displacements):
    """ 
    Given a graph, a list of the node indices to apply the displacements to, and a numpy array of displacements,
    update the graph.
    """
    for i in range(len(list_of_nodes)):
        graph.nodes[list_of_nodes[i]]['pos'] -= displacements[i]
    return graph

def propagate_displacements(graph, list_of_nodes, displacements, max_edge_length, min_edge_length, n_steps=1, max_iters=20):
    """
    """
    updated_graph = graph.copy()
    disp_step = displacements / n_steps
    # iterate through n_steps (we want to break the displacements into chunks to ensure motion makes sense)
    for step in range(n_steps):
        # apply the displacement to the graph nodes in the list_of_nodes
        updated_graph = apply_displacements_to_nodes(updated_graph, list_of_nodes, disp_step)
        # iterate through the iterations
        for _ in range(max_iters):
            # iterate through all the nodes in the graph
            for node in updated_graph.nodes:
                # get the neighbors of the current node
                neighbors = list(updated_graph.neighbors(node))
                # adjust the node positions to satisfy the edge length min/max constraints
                for neighbor in neighbors:
                    current_length = np.linalg.norm(np.array(updated_graph.nodes[node]['pos']) - np.array(updated_graph.nodes[neighbor]['pos']))
                    if current_length > max_edge_length:
                        dir = np.array(updated_graph.nodes[neighbor]['pos']) - np.array(updated_graph.nodes[node]['pos'])
                        dir /= np.linalg.norm(dir)
                        disp = (current_length - max_edge_length) * dir * 0.5

                        # update positions
                        updated_graph.nodes[node]['pos'] += disp
                        updated_graph.nodes[neighbor]['pos'] -= disp

                    elif current_length < min_edge_length:
                        dir = np.array(updated_graph.nodes[neighbor]['pos']) - np.array(updated_graph.nodes[node]['pos'])
                        dir /= np.linalg.norm(dir)
                        disp = (min_edge_length - current_length) * dir * 0.5

                        # update positions
                        updated_graph.nodes[node]['pos'] -= disp
                        updated_graph.nodes[neighbor]['pos'] += disp

        # check if the edge length constraints are satisfied
        edge_lengths = [(u, v, np.linalg.norm(np.array(updated_graph.nodes[u]['pos']) - np.array(updated_graph.nodes[v]['pos']))) for u, v in updated_graph.edges]
        if all(length <= max_edge_length for _, _, length in edge_lengths):
            break
    return updated_graph

def predict_centroid_dynamics(state, action):
    """
    Given a state and action, get the centroids and predict the next state geometrically.
    """
    # get state centroid
    state = torch.unsqueeze(state, 0).cuda()
    group_func = Group(num_group = 64, group_size = 32)
    _, centroids = group_func(state)
    centroids = centroids.squeeze().cpu().detach().numpy()
    state = centroids

    # sanity check point distances for graph dynamics
    kdtree = KDTree(state)
    nearest_dist, nearest_idx = kdtree.query(state, k=2)
    # max = np.amax(nearest_dist)
    # if max > 1:
    #     print("scaling...")
    # #     state = state / 10
    # print("Min dist: ", np.amin(nearest_dist))
    # print("Max dist: ", np.amax(nearest_dist))
    # print("mean dist: ", np.mean(nearest_dist))


    # state = state.detach().numpy()
    action = action.detach().numpy()

    # unnormalize the action
    action = unnormalize_a(action)

    # center the action at the origin of the point cloud
    # pcl_center = np.array([0.6, 0.0, 0.24]) # verified same pcl center that processed point clouds
    pcl_center = np.array([0.6, 0.0, 0.25])
    action[0:3] = action[0:3] - pcl_center
    action[0:3] = action[0:3] + np.array([0.005, -0.002, 0.0]) # observational correction

    # scale the action (multiply x,y,z,d by 10)
    action_scaled = action * 10
    action_scaled[3] = action[3] # don't scale the rotation
    len = 10 * 0.1 # 8cm scaled  to point cloud scaling # TODO: figure out grasp width scaling issue
    # len = 10 * 0.95
        
    # get the points and lines for the action orientation visualization
    ctr = action_scaled[0:3]
    upper_ctr = ctr + np.array([0,0, 0.6])
    rz = 90 + action_scaled[3]
    points, lines = line_3d_start_end(ctr, rz, len)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (lines.shape[0],1)))

    delta = 0.99 - action_scaled[4]  # TODO: maybe add 0.95*action_scaled[4] ?????
    # delta = 0.95 - action_scaled[4]
    end_pts, _ = line_3d_start_end(ctr, rz, len - delta)
    top_end_pts, _ = line_3d_start_end(upper_ctr, rz, len - delta)

    # get the top points for the grasp (given gripper finger height)
    top_points, _ = line_3d_start_end(upper_ctr, rz, len)

    # gripper 1 
    g1_base_start, _ = line_3d_start_end(points[0], rz+90, 0.18)
    g1_base_end, _ = line_3d_start_end(end_pts[0], rz+90, 0.18)
    g1_top_start, _ = line_3d_start_end(top_points[0], rz+90, 0.18)
    g1_top_end, _ = line_3d_start_end(top_end_pts[0], rz+90, 0.18)
    g1_points, _ = line_3d_point_set([g1_base_start, g1_base_end, g1_top_start, g1_top_end])

    # create oriented bounding box
    g1_test = o3d.geometry.OrientedBoundingBox()
    g1_bbox = g1_test.create_from_points(o3d.utility.Vector3dVector(g1_points))
    g1_idx= g1_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))

    # get the points in state pcl inside gripper 1
    # g1_idx = points_inside_rectangle(g1_points, None, state)
    inlier_pts = state.copy()

    # get the displacement vector for the gripper 1 base
    g1_dir_unit = dir_vec_from_points(end_pts[0], points[0])
    g1_displacement_vec = end_pts[0] - points[0]

    # apply the displacement vector to all the points in the state point cloud
    g1_diffs = np.tile(end_pts[0], (inlier_pts[g1_idx,:].shape[0],1)) - inlier_pts[g1_idx,:] 
    g1_diffs = np.linalg.norm(g1_diffs, axis=1)
    # inlier_pts[g1_idx,:] = inlier_pts[g1_idx,:] -  np.tile(g1_diffs, (3,1)).T * np.tile(g1_dir_unit, (inlier_pts[g1_idx,:].shape[0],1))

    # gripper 2
    g2_base_start, _ = line_3d_start_end(points[1], rz+90, 0.18)
    g2_base_end, _ = line_3d_start_end(end_pts[1], rz+90, 0.18)
    g2_top_start, _ = line_3d_start_end(top_points[1], rz+90, 0.18)
    g2_top_end, _ = line_3d_start_end(top_end_pts[1], rz+90, 0.18)
    g2_points, _ = line_3d_point_set([g2_base_start, g2_base_end, g2_top_start, g2_top_end])

    # create oriented bounding box
    g2_test = o3d.geometry.OrientedBoundingBox()
    g2_bbox = g2_test.create_from_points(o3d.utility.Vector3dVector(g2_points))
    g2_idx = g2_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))

    # get the points in state pcl inside gripper 1
    # g2_idx = points_inside_rectangle(g2_points, None, state)

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
    # inlier_pts[g2_idx,:] = inlier_pts[g2_idx,:] -  np.tile(g2_diffs, (3,1)).T * np.tile(g2_dir_unit, (inlier_pts[g2_idx,:].shape[0],1))

    # test graph propagation
    G = create_graph_from_point_cloud(inlier_pts, k=5)
    grasp_indices = g1_idx + g2_idx
    displacements = np.concatenate((np.tile(g1_diffs, (3,1)).T * np.tile(g1_dir_unit, (inlier_pts[g1_idx,:].shape[0],1)), 
                                    np.tile(g2_diffs, (3,1)).T * np.tile(g2_dir_unit, (inlier_pts[g2_idx,:].shape[0],1))))

    new_graph = propagate_displacements(G, grasp_indices, displacements, max_edge_length=0.13, min_edge_length=0.025)

    # Past Combos:
        # 0.125/0.0025 ---> CD = 0.01516
        # 0.15/0.005 ---> CD = 0.014595
        # 0.18/0.005 ---> CD = 0.01519
        # 0.14/0.005 ---> CD = 0.01477
        # 0.13/0.005 ---> CD = 0.01366
        # 0.12/0.005 ---> CD = 0.01466
        # 0.13/0.01 ---> CD = 0.1481
        # 0.13/0.05 ---> CD = 0.014146
    inlier_pts = create_point_cloud_from_graph(new_graph)

    return inlier_pts