import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
from scipy.spatial import cKDTree


class PointCloudPreprocessor:
    def __init__(
        self,
        device='cpu',
        voxel_size=0.2,
        sor_neighbors=20,
        sor_std_ratio=2.0,
        ror_nb_points=6,
        ror_radius=0.5,
        apply_voxel_downsample=True,
        apply_sor=True,
        apply_ror=True,
        apply_plane_removal=True,
        plane_distance_threshold=0.05,
        plane_ransac_n=3,
        plane_num_iterations=1000,
        ignore_preprocessing=False
    ):
        """
        Point Cloud Preprocessor with various options for downsampling, denoising, and normalization.
        """
        self.device = torch.device(device)
        self.voxel_size = voxel_size
        self.sor_neighbors = sor_neighbors
        self.sor_std_ratio = sor_std_ratio
        self.ror_nb_points = ror_nb_points
        self.ror_radius = ror_radius
        self.apply_voxel_downsample = apply_voxel_downsample
        self.apply_sor = apply_sor
        self.apply_ror = apply_ror
        self.apply_plane_removal = apply_plane_removal
        self.plane_distance_threshold = plane_distance_threshold
        self.plane_ransac_n = plane_ransac_n
        self.plane_num_iterations = plane_num_iterations
        self.ignore_preprocessing = ignore_preprocessing
        self.anchor_frame = None
        self.anchor_mean = None
        self.anchor_max_distance = None

    def voxel_downsample(self, point_cloud):
        if self.voxel_size is None:
            return point_cloud

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        downsampled_points = torch.tensor(np.asarray(downsampled_pcd.points), device=self.device, dtype=torch.float32)
        return downsampled_points

    def sor_outlier_removal(self, point_cloud):
        distances = torch.cdist(point_cloud, point_cloud, p=2)
        sorted_distances, _ = torch.topk(distances, self.sor_neighbors + 1, dim=-1, largest=False)
        mean_distances = sorted_distances[:, 1:].mean(dim=1)

        mean = mean_distances.mean()
        std_dev = mean_distances.std()
        threshold = mean + self.sor_std_ratio * std_dev

        mask = mean_distances <= threshold
        filtered_points = point_cloud[mask]
        return filtered_points

    def ror_outlier_removal(self, point_cloud):
        tree = cKDTree(point_cloud.cpu().numpy())
        neighbors = tree.query_ball_point(point_cloud.cpu().numpy(), r=self.ror_radius)
        mask = np.array([len(neigh) >= self.ror_nb_points for neigh in neighbors])
        return point_cloud[mask]

    def remove_plane(self, point_cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=self.plane_ransac_n,
            num_iterations=self.plane_num_iterations,
        )

        pcd_without_plane = pcd.select_by_index(inliers, invert=True)
        filtered_points = torch.tensor(np.asarray(pcd_without_plane.points), device=self.device, dtype=torch.float32)
        return filtered_points

    def normalize_with_anchor(self, point_cloud):
        points_centered = point_cloud - self.anchor_mean
        normalized_points = points_centered / self.anchor_max_distance
        return normalized_points

    def set_anchor_frame(self, anchor_frame):
        anchor_frame = self.preprocess(anchor_frame)
        self.anchor_frame = anchor_frame
        self.anchor_mean = torch.mean(anchor_frame, dim=0)
        self.anchor_max_distance = torch.max(torch.norm(anchor_frame - self.anchor_mean, dim=1))

    def preprocess(self, point_cloud):
        if self.apply_voxel_downsample:
            point_cloud = self.voxel_downsample(point_cloud)
        if self.apply_sor:
            point_cloud = self.sor_outlier_removal(point_cloud)
        if self.apply_ror:
            point_cloud = self.ror_outlier_removal(point_cloud)
        if self.apply_plane_removal:
            point_cloud = self.remove_plane(point_cloud)

        if self.anchor_mean is not None and self.anchor_max_distance is not None:
            point_cloud = self.normalize_with_anchor(point_cloud)

        return point_cloud

    def process_frame(self, frame):
        if self.anchor_frame is None:
            raise ValueError("Anchor frame is not set. Call `set_anchor_frame` first.")

        frame = self.preprocess(frame)
        return frame

    def process_folder(self, folder_path, start_frame=None, end_frame=None):
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])
        if start_frame is not None and end_frame is not None:
            files = files[start_frame:end_frame]
        elif start_frame is not None:
            files = files[start_frame:]
        elif end_frame is not None:
            files = files[:end_frame]
        else:
            files = files

        processed_sequence = []

        for idx, file in enumerate(tqdm(files, desc="Processing Files")):
            file_path = os.path.join(folder_path, file)
            pcd = o3d.io.read_point_cloud(file_path)
            frame = torch.tensor(np.asarray(pcd.points), device=self.device, dtype=torch.float32)

            if idx == 0:
                self.set_anchor_frame(frame)
            else:
                if self.ignore_preprocessing:
                    processed_frame = frame
                else:
                    processed_frame = self.process_frame(frame)
                processed_sequence.append(processed_frame)

        return processed_sequence
