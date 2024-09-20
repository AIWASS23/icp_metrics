import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os
import random
import torch
import open3d as o3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def extract_keypoints(points, voxel_size):
#     cloud = trimesh.points.PointCloud(points)
#     cloud = cloud.voxelized(voxel_size)
#     return cloud.points

def extract_keypoints(points, num_keypoints):
    if points.shape[0] > num_keypoints:
        indices = np.random.choice(points.shape[0], num_keypoints, replace=False)
        return points[indices]
    return points

def remove_outliers(points, threshold=1.5):
    mean = np.mean(points, axis=0)
    dist = np.linalg.norm(points - mean, axis=1)
    return points[dist < threshold * np.std(dist)]

def load_point_cloud(file_path):
    mesh = trimesh.load(file_path)
    return np.array(mesh.vertices)

def voxel_downsample(points, voxel_size):
    voxel_grid = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, indices = np.unique(voxel_grid, axis=0, return_index=True)
    return points[indices]

def load_all_point_clouds(q1_dir): # q1_dir é o caminho para KITTI-Sequence
    point_clouds = []

    for root, dirs, files in os.walk(q1_dir):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                print(f"Carregando arquivo: {file_path}")
                point_cloud = load_point_cloud(file_path)
                point_clouds.append(point_cloud)

    return point_clouds

def closest_point(src, dst):
    distances = torch.cdist(src, dst)
    indices = torch.argmin(distances, dim=1)
    return indices

def estimate_rigid_transform(A, B):
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)

    H = (A - centroid_A).T @ (B - centroid_B)
    U, S, Vt = torch.svd(H)
    R = Vt @ U.T

    if torch.det(R) < 0:
        Vt[:, -1] *= -1
        R = Vt @ U.T

    t = centroid_B - R @ centroid_A
    return R, t

def icp(src, dst, max_iterations=100, tolerance=1e-6):
    if src.shape[0] > dst.shape[0]:
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)]
    elif dst.shape[0] > src.shape[0]:
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)]

    src_homogeneous = np.ones((src.shape[0], 4))
    src_homogeneous[:, :-1] = src
    T = np.eye(4)
    prev_error = float('inf')

    for i in range(max_iterations):
        indices = closest_point(src_homogeneous[:, :-1], dst)
        matched_src = src_homogeneous[:, :-1][indices]

        R, t = estimate_rigid_transform(matched_src, dst)

        src_homogeneous[:, :-1] = (R @ src_homogeneous[:, :-1].T).T + t

        error = np.mean(np.linalg.norm(src_homogeneous[:, :-1] - dst, axis=1))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    return T


def icp_with_voxel(src, dst, max_iterations=200, tolerance=1e-6):
    src = voxel_downsample(src, 0.2)  # Ajuste o valor do voxel_size conforme necessário
    dst = voxel_downsample(dst, 0.2)

    # Garantir que as nuvens subamostradas tenham o mesmo número de pontos
    min_points = min(src.shape[0], dst.shape[0])
    src = src[:min_points]
    dst = dst[:min_points]

    src_homogeneous = np.ones((src.shape[0], 4))
    src_homogeneous[:, :-1] = src

    T = np.eye(4)
    prev_error = float('inf')

    # Construir a árvore KD apenas uma vez
    tree = KDTree(dst)

    for i in range(max_iterations):
        distances, indices = tree.query(src_homogeneous[:, :-1])

        # Corrigindo os índices para garantir que não ultrapassem o tamanho do array
        indices = np.clip(indices, 0, dst.shape[0] - 1)

        matched_src = src_homogeneous[:, :-1][indices]

        R, t = estimate_rigid_transform(matched_src, dst)

        src_homogeneous[:, :-1] = (R @ src_homogeneous[:, :-1].T).T + t

        error = np.mean(np.linalg.norm(src_homogeneous[:, :-1] - dst, axis=1))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    return T

def icp_torch(src, dst, max_iterations=300, tolerance=1e-6): # Quanto maior o numero de iteracoes mais preciso é o desenho do caminho

    #src = extract_keypoints(src, 500)  # Extrair 500 keypoints
    #dst = extract_keypoints(dst, 500)

    src = torch.tensor(src, dtype=torch.float32, device=device)
    dst = torch.tensor(dst, dtype=torch.float32, device=device)

    for i in range(max_iterations):
        indices = closest_point(src, dst)
        matched_dst = dst[indices]

        R, t = estimate_rigid_transform(src, matched_dst)
        src = (R @ src.T).T + t

        error = torch.mean(torch.norm(src - matched_dst, dim=1))
        if error < tolerance:
            break

        # Limpar a memória da GPU
        torch.cuda.empty_cache()

    return R.cpu().numpy(), t.cpu().numpy()

# def estimate_trajectory(scans):
#     trajectory = [np.eye(4)]  # A trajetória começa com a identidade (ponto inicial)

#     for i in range(1, len(scans)):
#         T = icp(scans[i-1], scans[i])
#         trajectory.append(trajectory[-1] @ T)

#     return np.array(trajectory)

# def estimate_trajectory(scans):
#     trajectory = [np.eye(4)]
#     for i in range(1, len(scans)):
#         R, t = icp_torch(scans[i-1], scans[i])
#         T = np.eye(4)
#         T[:3, :3] = R
#         T[:3, 3] = t
#         trajectory.append(trajectory[-1] @ T)
#     return np.array(trajectory)

def estimate_trajectory(scans):
    trajectory = [np.eye(4)]
    for i in range(1, len(scans)):
        src = voxel_downsample(scans[i-1], 0.3)
        dst = voxel_downsample(scans[i], 0.3)

        R, t = icp_torch(src, dst)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        trajectory.append(trajectory[-1] @ T)
    return np.array(trajectory)

def compare_with_ground_truth(estimated_trajectory, ground_truth):
    error = np.linalg.norm(estimated_trajectory - ground_truth, axis=(1, 2))
    print("Erro médio: ", np.mean(error))
    return error

def plot_trajectory(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extraindo as coordenadas (x, y, z)
    x = trajectory[:, 0, 3]
    y = trajectory[:, 1, 3]
    z = trajectory[:, 2, 3]

    ax.plot(x, y, z, label='Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def calculate_ground_truth_trajectory(scans):
    trajectory = [np.eye(4)]  # Começa com a identidade

    pcd_prev = o3d.geometry.PointCloud()
    pcd_prev.points = o3d.utility.Vector3dVector(scans[0])

    for i in range(1, len(scans)):
        pcd_curr = o3d.geometry.PointCloud()
        pcd_curr.points = o3d.utility.Vector3dVector(scans[i])

        reg_icp = o3d.pipelines.registration.registration_icp(
            pcd_curr, pcd_prev,
            max_correspondence_distance=0.2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        T = reg_icp.transformation
        trajectory.append(trajectory[-1] @ T)

        pcd_prev = pcd_curr

    return np.array(trajectory)

def plot_comparison(estimated, reference):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trajetória estimada
    ax.plot(estimated[:, 0, 3], estimated[:, 1, 3], estimated[:, 2, 3], label='Trajectory Estimada', color='blue')

    # Trajetória de referência
    ax.plot(reference[:, 0, 3], reference[:, 1, 3], reference[:, 2, 3], label='Trajectory de Referência', color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def compare_trajectories(estimated, reference):
    error = np.linalg.norm(estimated - reference, axis=(1, 2))
    mean_error = np.mean(error)
    print("Erro médio em comparação com a open3d: ", mean_error)
    return error


ground_truth = np.load('ground_truth.npy')

scans = load_all_point_clouds(q1_dir)

estimated_trajectory = estimate_trajectory(scans)
reference_trajectory = calculate_ground_truth_trajectory(scans)

compare_with_ground_truth(estimated_trajectory, ground_truth)

compare_trajectories(estimated_trajectory, reference_trajectory)

plot_trajectory(estimated_trajectory)
plot_comparison(estimated_trajectory, reference_trajectory)