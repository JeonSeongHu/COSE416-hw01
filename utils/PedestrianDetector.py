# utils/pedestrian_detector.py

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class PedestrianDetector:
    def __init__(self, eps=0.5, min_samples=10, movement_threshold=0.5, displacement_threshold=0.5, decay_rate=0.9, device='cpu'):
        """
        클러스터링 기반 움직이는 객체 검출기 초기화.

        Parameters:
        - eps (float): DBSCAN의 epsilon 파라미터.
        - min_samples (int): DBSCAN의 최소 샘플 수.
        - movement_threshold (float): 누적 이동 거리가 이 값을 넘으면 움직임으로 간주.
        - displacement_threshold (float): 변위가 이 값을 넘으면 움직임으로 간주.
        - decay_rate (float): 잔차가 시간에 따라 감쇠하는 비율 (0과 1 사이).
        - device (str): 'cpu' 또는 'cuda'.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.movement_threshold = movement_threshold
        self.displacement_threshold = displacement_threshold
        self.decay_rate = decay_rate
        self.device = torch.device(device)
        self.cluster_dict = {}  # 클러스터 ID별 정보 저장
        self.next_cluster_id = 0  # 새로운 클러스터 ID 생성용

    def cluster_frame(self, frame_points):
        """
        프레임의 포인트 클라우드를 클러스터링합니다.

        Parameters:
        - frame_points (torch.Tensor): (N, 3) 형태의 포인트 클라우드.

        Returns:
        - clusters (list of np.ndarray): 클러스터 포인트들의 리스트.
        - centroids (np.ndarray): 각 클러스터의 중심점 배열.
        - labels (np.ndarray): 각 포인트의 클러스터 레이블 배열.
        """
        points_np = frame_points.cpu().numpy()
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_np)
        labels = clustering.labels_

        unique_labels = set(labels)
        clusters = []
        centroids = []

        for label in unique_labels:
            if label == -1:
                continue  # 노이즈로 간주
            class_member_mask = (labels == label)
            cluster = points_np[class_member_mask]
            clusters.append(cluster)
            centroid = cluster.mean(axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)
        return clusters, centroids, labels

    def match_and_update_clusters(self, clusters, centroids):
        """
        현재 프레임의 클러스터를 이전 클러스터와 매칭하고 업데이트합니다.

        Parameters:
        - clusters (list of np.ndarray): 현재 프레임의 클러스터 리스트.
        - centroids (np.ndarray): 현재 프레임의 클러스터 중심점들.
        """
        current_centroids = centroids
        if not self.cluster_dict:
            # 첫 프레임인 경우 클러스터 초기화
            for idx, (centroid, cluster) in enumerate(zip(current_centroids, clusters)):
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                self.cluster_dict[cluster_id] = {
                    'initial_centroid': centroid,
                    'centroid': centroid,
                    'cumulative_residual': 0.0,
                    'cluster': cluster,
                    'missed_frames': 0
                }
        else:
            existing_centroids = np.array([info['centroid'] for info in self.cluster_dict.values()])
            existing_ids = list(self.cluster_dict.keys())

            # 기존 클러스터와 현재 클러스터 간 거리 매트릭스 계산
            distance_matrix = cdist(existing_centroids, current_centroids)

            # 최대 매칭 거리 설정
            max_distance = self.eps * 2
            cost_matrix = np.copy(distance_matrix)
            cost_matrix[cost_matrix > max_distance] = max_distance * 10  # 큰 값으로 대체

            # 헝가리안 알고리즘을 사용한 최적 매칭 계산
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_current_idxs = set()
            assigned_existing_ids = set()
            for existing_idx, current_idx in zip(row_ind, col_ind):
                distance = distance_matrix[existing_idx, current_idx]
                if distance < max_distance:
                    # 매칭된 경우
                    cluster_id = existing_ids[existing_idx]
                    movement = np.linalg.norm(current_centroids[current_idx] - existing_centroids[existing_idx])

                    # 프레임별 움직임 임계값 적용하여 노이즈 무시
                    if movement > self.eps * 0.1:
                        # 감쇠를 적용하여 누적 잔차 업데이트
                        self.cluster_dict[cluster_id]['cumulative_residual'] = (
                            self.cluster_dict[cluster_id]['cumulative_residual'] * self.decay_rate + movement
                        )
                    else:
                        # 움직임이 미미한 경우 감쇠만 적용
                        self.cluster_dict[cluster_id]['cumulative_residual'] *= self.decay_rate

                    # 중심점과 클러스터 포인트 업데이트
                    self.cluster_dict[cluster_id]['centroid'] = current_centroids[current_idx]
                    self.cluster_dict[cluster_id]['cluster'] = clusters[current_idx]
                    self.cluster_dict[cluster_id]['missed_frames'] = 0
                    assigned_current_idxs.add(current_idx)
                    assigned_existing_ids.add(cluster_id)
                else:
                    # 유효한 매칭이 없는 경우
                    pass

            # 새로운 클러스터 추가
            for idx, (centroid, cluster) in enumerate(zip(current_centroids, clusters)):
                if idx not in assigned_current_idxs:
                    cluster_id = self.next_cluster_id
                    self.next_cluster_id += 1
                    self.cluster_dict[cluster_id] = {
                        'initial_centroid': centroid,
                        'centroid': centroid,
                        'cumulative_residual': 0.0,
                        'cluster': cluster,
                        'missed_frames': 0
                    }

            # 사라진 클러스터 처리
            for cluster_id in existing_ids:
                if cluster_id not in assigned_existing_ids:
                    self.cluster_dict[cluster_id]['missed_frames'] += 1
                    # 몇 프레임 동안 보이지 않으면 제거
                    if self.cluster_dict[cluster_id]['missed_frames'] > 5:
                        del self.cluster_dict[cluster_id]

    def detect(self, current_frame):
        """
        현재 프레임에서 움직이는 객체를 검출합니다.

        Parameters:
        - current_frame (torch.Tensor): (N, 3) 형태의 현재 프레임 포인트 클라우드.

        Returns:
        - moving_clusters (list of np.ndarray): 움직이는 클러스터들의 포인트 리스트.
        - residual_list (list of dict): 각 움직이는 클러스터의 누적 잔차 및 변위 정보 리스트.
        """
        # 현재 프레임 클러스터링
        clusters, centroids, labels = self.cluster_frame(current_frame)

        # 클러스터 매칭 및 업데이트
        self.match_and_update_clusters(clusters, centroids)

        moving_clusters = []
        residual_list = []

        for cluster_id, cluster_info in self.cluster_dict.items():
            cumulative_residual = cluster_info['cumulative_residual']
            missed_frames = cluster_info['missed_frames']
            displacement = np.linalg.norm(cluster_info['centroid'] - cluster_info['initial_centroid'])
            
            # print(f"Cluster {cluster_id}: residual={cumulative_residual:.2f}, displacement={displacement:.2f}, missed_frames={missed_frames}")
            # 누적 이동 거리와 변위를 모두 고려하여 움직임 판단
            if (cumulative_residual > self.movement_threshold and displacement > self.displacement_threshold) and missed_frames == 0:
                moving_clusters.append(cluster_info['cluster'])
                residual_list.append({
                    'cluster_id': cluster_id,
                    'cumulative_residual': cumulative_residual,
                    'displacement': displacement
                })

        return moving_clusters, residual_list
1