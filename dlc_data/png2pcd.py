from PIL import Image
import numpy as np
import open3d as o3d
import random

def load_depth_image(jpg_path):
    """
    从 JPG 文件加载深度图。
    :param jpg_path: 深度图 JPG 文件的路径。
    :return: 深度图的 NumPy 数组。
    """
    with Image.open(jpg_path) as img:
        depth_image = np.array(img)
    return depth_image

def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    """
    将深度图像转换为点云。
    :param depth_image: 深度图像的 NumPy 数组。
    :param camera_intrinsics: 相机的内参矩阵。
    :return: Open3D 点云对象。
    """
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width=depth_image.shape[1], height=depth_image.shape[0],
                              fx=camera_intrinsics['fx'], fy=camera_intrinsics['fy'],
                              cx=camera_intrinsics['cx'], cy=camera_intrinsics['cy'])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics)
    return pcd

def adjust_point_cloud(pcd, target_points=1024):
    """
    调整点云，确保它有 target_points 个点。
    :param pcd: Open3D 点云对象。
    :param target_points: 目标点数。
    :return: 调整后的 Open3D 点云对象。
    """
    current_points = np.asarray(pcd.points)
    if len(current_points) > target_points:
        # 如果点数超过目标点数，则随机删除一些点
        chosen_indices = random.sample(range(len(current_points)), target_points)
        new_points = current_points[chosen_indices]
    elif len(current_points) < target_points:
        # 如果点数少于目标点数，则从原始点云中随机添加点
        additional_indices = random.choices(range(len(current_points)), k=target_points - len(current_points))
        new_points = np.vstack((current_points, current_points[additional_indices]))
    else:
        # 如果点数正好等于目标点数，则不做改变
        new_points = current_points

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    return new_pcd

# 加载深度图
depth_image_path = '00101.png'
depth_image = load_depth_image(depth_image_path)

# 设置相机内参
camera_intrinsics = {
    'fx': 913.492004,
    'fy': 913.715515,
    'cx': 959.184265,
    'cy': 545.431702
}

# 转换深度图为点云
point_cloud = depth_image_to_point_cloud(depth_image, camera_intrinsics)

# 调整点云以确保点数为1024
adjusted_point_cloud = adjust_point_cloud(point_cloud, 1024)

# 可视化和保存
o3d.visualization.draw_geometries([point_cloud])
o3d.visualization.draw_geometries([adjusted_point_cloud])

o3d.io.write_point_cloud("00101.ply", point_cloud)
o3d.io.write_point_cloud("00101_1024.ply", adjusted_point_cloud)
