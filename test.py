import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Função para carregar as nuvens de pontos
def load_point_cloud(file_path):
    mesh = trimesh.load(file_path)
    return np.array(mesh.vertices)

# Exemplo de uso
scan_path = "000000_points.obj"
point_cloud = load_point_cloud(scan_path)


# Encontrar o ponto mais próximo na nuvem de destino para cada ponto na nuvem de origem
def closest_point(src, dst):
    tree = KDTree(dst)
    distances, indices = tree.query(src)
    return indices

def icp(src, dst, max_iterations=100, tolerance=1e-6):
    src_homogeneous = np.ones((src.shape[0], 4))
    src_homogeneous[:, :-1] = src
    T = np.eye(4)
    prev_error = float('inf')

    for i in range(max_iterations):
        indices = closest_point(src_homogeneous[:, :-1], dst)
        matched_src = src_homogeneous[:, :-1][indices]
        
        # Estimar a transformação (R, t)
        R, t = estimate_rigid_transform(matched_src, dst)

        # Atualizar a nuvem de pontos fonte com a nova transformação
        src_homogeneous[:, :-1] = (R @ src_homogeneous[:, :-1].T).T + t

        # Calcular o erro médio
        error = np.mean(np.linalg.norm(src_homogeneous[:, :-1] - dst, axis=1))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    return T


def estimate_rigid_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    H = (A - centroid_A).T @ (B - centroid_B)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Garantir que a rotação seja válida
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B.T - R @ centroid_A.T

    return R, t


def estimate_trajectory(scans):
    trajectory = [np.eye(4)]  # A trajetória começa com a identidade (ponto inicial)

    for i in range(1, len(scans)):
        T = icp(scans[i-1], scans[i])
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


# Carregando ground-truth
ground_truth = np.load('ground_truth.npy')

# Estimando a trajetória
scans = [load_point_cloud(f"scan_{i:06d}.obj") for i in range(30)]
estimated_trajectory = estimate_trajectory(scans)

# Comparando com a ground-truth
compare_with_ground_truth(estimated_trajectory, ground_truth)

# Plotando a trajetória
plot_trajectory(estimated_trajectory)
