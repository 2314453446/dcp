import torch
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import DCP
from util import transform_point_cloud

def load_point_cloud(file_path):
    """
    从文件加载点云数据。
    :param file_path: 点云数据文件的路径。
    :return: 点云的 NumPy 数组。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)



def visualize_point_cloud(points, title="Point Cloud", color='blue'):
    """
    Visualize a single point cloud using matplotlib.
    points: N x 3 numpy array, where N is the number of points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.5, s=3)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)
    plt.show()

def visualize_point_cloud_trans(points, title="Point Cloud", color='blue'):
    """
    Visualize a single point cloud using matplotlib.
    points: 3 x N numpy array, where N is the number of points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points_transposed = points.T
    ax.scatter(points_transposed[:, 0], points_transposed[:, 1], points_transposed[:, 2], color=color, alpha=0.5, s=5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)
    plt.show()

def visualize_combined_point_cloud(points1, points2, title="Combined Point Cloud"):
    """
    Visualize two point clouds in a single plot using matplotlib.
    points1, points2: N x 3 numpy arrays, where N is the number of points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='blue', alpha=0.5, s=3)
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='red', alpha=0.5, s=3)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)
    plt.show()

def main():
    # 定义模型参数
    args = argparse.Namespace(emb_nn='dgcnn', pointer='identity', head='svd', emb_dims=512, cycle=False)

    # 加载模型
    model = DCP(args).cuda()
    model.load_state_dict(torch.load('/home/tuolong/learning/dcp/pretrained/dcp_v1.t7'))  # 替换为预训练模型的路径
    model.eval()

    # 加载自己的点云数据
    src = load_point_cloud('/home/tuolong/learning/dcp/dlc_data/00100_1024.ply')  # 替换为您的点云数据文件路径
    target = load_point_cloud('/home/tuolong/learning/dcp/dlc_data/00101_1024.ply')  # 替换为您的点云数据文件路径

    # 确保点云具有正确的形状和类型
    src = torch.tensor(src[:1024], dtype=torch.float).unsqueeze(0).cuda() # [1,1024,3]
    target = torch.tensor(target[:1024], dtype=torch.float).unsqueeze(0).cuda() # [1,1024,3]
    src = src.transpose(1, 2)
    target = target.transpose(1, 2)
    print(src.shape)

    # 运行模型，匹配两个点云
    rotation_ab, translation_ab, _, _ = model(src, target)

    # 将点云从 GPU 转移到 CPU 并转换为 NumPy 数组
    src = src.cpu().squeeze().numpy()
    target = target.cpu().squeeze().numpy()
    transformed_src = transform_point_cloud(torch.tensor(src).cuda(), rotation_ab, translation_ab).detach().cpu().numpy()

    # 同时在一个新窗口中展示所有点云
    # 同时在一个新窗口中展示所有点云
    visualize_combined_point_cloud(transformed_src, target.T,
                                   title="Combined Point Clouds - Red: Transformed Source, Green: Target")
    visualize_combined_point_cloud(src.T, target.T, title="Combined Point Clouds - Blue: Source, Green: Target")


if __name__ == "__main__":
    main()
