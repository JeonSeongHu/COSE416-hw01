# utils/pedestrian_detector.py

import torch
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
import logging

class PedestrianDetector:
    def __init__(self, eps=0.5, min_samples=10, device='cpu'):
        """
        보행자 검출기 초기화.

        Parameters:
        - eps (float): DBSCAN의 epsilon 파라미터.
        - min_samples (int): DBSCAN의 최소 샘플 수.
        - device (str): 'cpu' 또는 'cuda'.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.device = torch.device(device)

    def compute_residuals(self, source, target):
        """
        두 프레임 간의 잔차를 계산합니다.

        Parameters:
        - source (torch.Tensor): (N, 3) 형태의 현재 프레임 포인트 클라우드.
        - target (torch.Tensor): (N, 3) 형태의 이전 프레임 포인트 클라우드.

        Returns:
        - residuals (torch.Tensor): (N, 3) 형태의 잔차 포인트 클라우드.
        """
        residuals = source - target
        return residuals

    def filter_moving_points(self, residuals, threshold=0.05):
        """
        노이즈를 제거하고 움직이는 점들을 필터링합니다.

        Parameters:
        - residuals (torch.Tensor): (N, 3) 형태의 잔차 포인트 클라우드.
        - threshold (float): 움직임 임계값.

        Returns:
        - moving_points (torch.Tensor): (M, 3) 형태의 움직이는 점들.
        """
        residual_distances = torch.norm(residuals, dim=1)
        moving_mask = residual_distances > threshold
        moving_points = residuals[moving_mask]
        return moving_points

    def cluster_moving_points(self, moving_points):
        """
        움직이는 점들을 클러스터링하여 보행자를 검출합니다.

        Parameters:
        - moving_points (torch.Tensor): (M, 3) 형태의 움직이는 점들.

        Returns:
        - clusters (list of np.ndarray): 각 보행자에 해당하는 포인트들의 리스트.
        """
        if moving_points.shape[0] == 0:
            return []

        moving_points_np = moving_points.cpu().numpy()
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(moving_points_np)
        labels = clustering.labels_

        unique_labels = set(labels)
        clusters = []
        for label in unique_labels:
            if label == -1:
                continue  # 노이즈로 간주
            class_member_mask = (labels == label)
            cluster = moving_points_np[class_member_mask]
            clusters.append(cluster)

        return clusters

    def detect(self, current_frame, previous_frame):
        """
        두 프레임 간의 보행자를 검출합니다.

        Parameters:
        - current_frame (torch.Tensor): (N, 3) 형태의 현재 프레임 포인트 클라우드.
        - previous_frame (torch.Tensor): (N, 3) 형태의 이전 프레임 포인트 클라우드.

        Returns:
        - clusters (list of np.ndarray): 검출된 보행자 클러스터 리스트.
        """
        residuals = self.compute_residuals(current_frame, previous_frame)
        moving_points = self.filter_moving_points(residuals)
        clusters = self.cluster_moving_points(moving_points)
        return clusters
