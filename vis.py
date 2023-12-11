import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data import ModelNet40
from model import DCP
from torch.utils.data import DataLoader
from util import transform_point_cloud

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

    # 加载数据集
    dataset = ModelNet40(num_points=1024, partition='test')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)  # 加载两个样本

    for data in data_loader:
        src, target, _, _, _, _, _, _ = data
        src, target = src.cuda(), target.cuda()
        print(src.shape)
        # 运行模型，匹配两个点云
        rotation_ab, translation_ab, _, _ = model(src, target)

        # 将点云从 GPU 转移到 CPU 并转换为 NumPy 数组
        src = src.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()
        transformed_src = transform_point_cloud(torch.tensor(src).cuda(), rotation_ab, translation_ab).detach().cpu().numpy()

        # # 使用原有的可视化方法
        # visualize_point_cloud_trans(src, title="Source Point Cloud", color='blue')
        # visualize_point_cloud(target, title="Target Point Cloud", color='green')
        # visualize_point_cloud(transformed_src, title="Transformed Source Point Cloud", color='red')

        # 同时在一个新窗口中展示所有点云
        visualize_combined_point_cloud(src.T, target, title="Combined Point Clouds - Blue: Source, Green: Target")
        visualize_combined_point_cloud(transformed_src, target, title="Combined Point Clouds - Red: Transformed Source, Green: Target")

        break  # 只处理第一对样本

if __name__ == "__main__":
    main()
