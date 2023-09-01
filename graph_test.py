import torch.nn as nn
import torch
import numpy as np
import networkx as nx
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from pointnet2_ops import pointnet2_utils


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
    return pos

def propagate_dispacements(graph, node_to_move, displacement, max_edge_length, min_edge_length, max_iterations=100):
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
                


# 3d graph
state = np.expand_dims(np.load('/home/alison/Clay_Data/Fully_Processed/Aug29_Human_Demos/States/shell_scaled_state0.npy'), axis=0)
state = torch.from_numpy(state).cuda()
# get state centroid
group_func = Group(num_group = 64, group_size = 32)
_, centroids = group_func(state)
centroids = centroids.squeeze().cpu().detach().numpy()

# arr = np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]])
# G = nx.cycle_graph(20)
G = create_graph_from_point_cloud(centroids, k=5)

ctroid = create_point_cloud_from_graph(G)
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(ctroid)
pcl_colors = np.tile(np.array([1, 0, 0]), (ctroid.shape[0],1))
pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
# o3d.visualization.draw_geometries([pcl])

new_graph = propagate_dispacements(G, 1, np.array([0.2, 0.5, 0.0]), 0.5, 0.08)

ns_ctroid = create_point_cloud_from_graph(new_graph)
ns_pcl = o3d.geometry.PointCloud()
ns_pcl.points = o3d.utility.Vector3dVector(ns_ctroid)
ns_pcl_colors = np.tile(np.array([0, 0, 1]), (ns_ctroid.shape[0],1))
ns_pcl.colors = o3d.utility.Vector3dVector(ns_pcl_colors)
o3d.visualization.draw_geometries([pcl, ns_pcl])

# 3d spring layout
pos = nx.spring_layout(G, dim=3, seed=789)
# extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# create 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the nodes
ax.scatter(*node_xyz.T, s=100, marker='s', color='r')

# plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, linewidth=3, color='b')

# format axes
ax.grid(False)
for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    dim.set_ticks([])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig.tight_layout()
plt.show()