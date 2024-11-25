import torch
import numpy as np
import open3d as o3d
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


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
    ):
        """
        Point Cloud Preprocessor.

        Parameters:
        - device (str): 'cpu' 또는 'cuda' (GPU 사용 시 'cuda').
        - voxel_size (float): Voxel downsampling 크기 (None이면 사용하지 않음).
        - sor_neighbors (int): SOR에서 고려할 이웃 포인트 수.
        - sor_std_ratio (float): SOR에서 Outlier 판별 기준.
        - ror_nb_points (int): ROR에서 고려할 최소 이웃 포인트 수.
        - ror_radius (float): ROR에서 이웃 탐색 반경.
        - apply_voxel_downsample (bool): Voxel downsampling 적용 여부.
        - apply_sor (bool): SOR 적용 여부.
        - apply_ror (bool): ROR 적용 여부.
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
        self.anchor_frame = None  # Anchor frame 저장
        self.anchor_transform = None  # Anchor frame의 좌표계
        self.anchor_mean = None  # Anchor frame의 mean 저장
        self.anchor_max_distance = None  # Anchor frame의 max_distance 저장

    def voxel_downsample(self, point_cloud):
        if self.voxel_size is None:
            return point_cloud

        logging.info(f"Applying voxel downsampling with voxel size: {self.voxel_size}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        downsampled_points = torch.tensor(np.asarray(downsampled_pcd.points), device=self.device, dtype=torch.float32)
        logging.info(f"Voxel downsampling complete. Points reduced from {point_cloud.shape[0]} to {downsampled_points.shape[0]}")
        return downsampled_points

    def sor_outlier_removal(self, point_cloud):
        logging.info(f"Applying SOR with {self.sor_neighbors} neighbors and {self.sor_std_ratio} std ratio")
        distances = torch.cdist(point_cloud, point_cloud, p=2)
        sorted_distances, _ = torch.topk(distances, self.sor_neighbors + 1, dim=-1, largest=False)
        mean_distances = sorted_distances[:, 1:].mean(dim=1)

        mean = mean_distances.mean()
        std_dev = mean_distances.std()
        threshold = mean + self.sor_std_ratio * std_dev

        mask = mean_distances <= threshold
        filtered_points = point_cloud[mask]
        logging.info(f"SOR complete. Points reduced from {point_cloud.shape[0]} to {filtered_points.shape[0]}")
        return filtered_points

    def ror_outlier_removal(self, point_cloud):
        logging.info(f"Applying ROR with radius {self.ror_radius} and min points {self.ror_nb_points}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
        cl, ind = pcd.remove_radius_outlier(nb_points=self.ror_nb_points, radius=self.ror_radius)
        filtered_pcd = pcd.select_by_index(ind)
        filtered_points = torch.tensor(np.asarray(filtered_pcd.points), device=self.device, dtype=torch.float32)
        logging.info(f"ROR complete. Points reduced from {point_cloud.shape[0]} to {filtered_points.shape[0]}")
        return filtered_points

    def normalize_with_anchor(self, point_cloud):
        if self.anchor_mean is None or self.anchor_max_distance is None:
            raise ValueError("Anchor frame must be set before normalizing with anchor frame.")

        logging.info(f"Normalizing point cloud with anchor frame's mean and max distance.")
        points_centered = point_cloud - self.anchor_mean
        normalized_points = points_centered / self.anchor_max_distance
        logging.info("Anchor-based normalization complete.")
        return normalized_points

    def set_anchor_frame(self, anchor_frame):
        logging.info("Setting anchor frame.")
        anchor_frame = self.preprocess(anchor_frame)
        self.anchor_frame = anchor_frame
        self.anchor_mean = torch.mean(anchor_frame, dim=0)
        self.anchor_max_distance = torch.max(torch.norm(anchor_frame - self.anchor_mean, dim=1))
        self.anchor_transform = torch.eye(4, device=self.device)

    def register_frames(self, source, target):
        logging.info(f"Registering source (size: {source.shape[0]}) with target (size: {target.shape[0]})")
        source_o3d = o3d.geometry.PointCloud()
        source_o3d.points = o3d.utility.Vector3dVector(source.cpu().numpy())
        target_o3d = o3d.geometry.PointCloud()
        target_o3d.points = o3d.utility.Vector3dVector(target.cpu().numpy())

        reg_result = o3d.pipelines.registration.registration_icp(
            source_o3d,
            target_o3d,
            max_correspondence_distance=1.0,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        transform = torch.tensor(reg_result.transformation, device=self.device, dtype=torch.float32)
        source_homo = torch.cat([source, torch.ones((source.shape[0], 1), device=self.device)], dim=1)
        registered_source = (transform @ source_homo.T).T[:, :3]

        logging.info("Registration complete.")
        return registered_source, transform

    def preprocess(self, point_cloud):
        if self.apply_voxel_downsample:
            point_cloud = self.voxel_downsample(point_cloud)
        if self.apply_sor:
            point_cloud = self.sor_outlier_removal(point_cloud)
        if self.apply_ror:
            point_cloud = self.ror_outlier_removal(point_cloud)

        if self.anchor_mean is not None and self.anchor_max_distance is not None:
            point_cloud = self.normalize_with_anchor(point_cloud)
        else:
            logging.warning("Anchor frame not set. Skipping normalization.")

        return point_cloud

    def process_frame(self, frame):
        if self.anchor_frame is None:
            raise ValueError("Anchor frame is not set. Call `set_anchor_frame` first.")
        
        logging.info("Processing frame against anchor frame.")
        frame = self.preprocess(frame)
        # registered_frame, _ = self.register_frames(frame, self.anchor_frame)
        # return registered_frame
        return frame

    def process_folder(self, folder_path, num = None):
        logging.info(f"Processing folder: {folder_path}")

        if num is not None:
            logging.info(f"Limiting number of files to process: {num}")
        else:
            logging.info("Processing all files in the folder.")
            num = len(os.listdir(folder_path))

        pcd_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')][1:num])
        processed_sequence = []

        for idx, file in enumerate(pcd_files):
            file_path = os.path.join(folder_path, file)
            logging.info(f"Reading file: {file_path}")
            pcd = o3d.io.read_point_cloud(file_path)
            frame = torch.tensor(np.asarray(pcd.points), device=self.device, dtype=torch.float32)

            if idx == 0:
                self.set_anchor_frame(frame)
            else:
                processed_frame = self.process_frame(frame)
                processed_sequence.append(processed_frame)

        logging.info("Folder processing complete.")
        return processed_sequence
