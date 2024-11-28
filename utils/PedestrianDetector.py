import torch
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

class PedestrianDetector:
    def __init__(self, eps=0.5, min_samples=10, max_samples=20, movement_threshold=0.5, 
                 displacement_threshold=0.5, decay_rate=0.9, device='cpu', max_cluster_size=0.1,
                 missed_frames_threshold=10):
        self.eps = eps
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.movement_threshold = movement_threshold
        self.displacement_threshold = displacement_threshold
        self.decay_rate = decay_rate
        self.device = torch.device(device)
        self.cluster_dict = {}  # Store cluster information by ID
        self.next_cluster_id = 0  # Unique ID for new clusters
        self.max_cluster_size = max_cluster_size
        self.existing_tree = None  # KD-Tree for existing clusters
        self.missed_frames_threshold = missed_frames_threshold

    def update_tree(self):
        """
        Update the KD-Tree with current cluster centroids.
        """
        if self.cluster_dict:
            existing_centroids = np.array([info['centroid'] for info in self.cluster_dict.values()])
            self.existing_tree = cKDTree(existing_centroids)
        else:
            self.existing_tree = None

    def cluster_frame(self, frame_points):
        points_np = frame_points.cpu().numpy()
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_np)
        labels = clustering.labels_

        unique_labels = set(labels)
        clusters = []
        centroids = []

        for label in unique_labels:
            if label == -1:
                continue  # Noise
            class_member_mask = (labels == label)
            cluster = points_np[class_member_mask]
            if len(cluster) > self.max_samples:
                continue
            elif np.any(np.max(cluster, axis=0) - np.min(cluster, axis=0) > self.max_cluster_size):
                continue
            clusters.append(cluster)
            centroids.append(cluster.mean(axis=0))

        centroids = np.array(centroids)
        return clusters, centroids, labels

    def match_and_update_clusters(self, clusters, centroids):
        current_centroids = centroids

        if not self.cluster_dict:
            self.initialize_clusters(current_centroids, clusters)
        else:
            max_distance = self.eps * 2
            assigned_current_idxs, assigned_existing_ids = self.match_clusters(current_centroids, clusters, max_distance)

            self.register_new_clusters(current_centroids, clusters, assigned_current_idxs)
            self.remove_missing_clusters(assigned_existing_ids)

        self.update_tree()

    def register_new_clusters(self, current_centroids, clusters, assigned_current_idxs):
        for idx, (centroid, cluster) in enumerate(zip(current_centroids, clusters)):
            if idx not in assigned_current_idxs:
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                self.cluster_dict[cluster_id] = {
                    'initial_centroid': centroid,
                    'centroid': centroid,
                    'cumulative_residual': self.movement_threshold // 2,
                    'cluster': cluster,
                    'missed_frames': 0
                }

    def remove_missing_clusters(self, assigned_existing_ids):
        for cluster_id in list(self.cluster_dict.keys()):
            if cluster_id not in assigned_existing_ids:
                self.cluster_dict[cluster_id]['missed_frames'] += 1
                if self.cluster_dict[cluster_id]['missed_frames'] > self.missed_frames_threshold:
                    del self.cluster_dict[cluster_id]


    def initialize_clusters(self, centroids, clusters):
        for centroid, cluster in zip(centroids, clusters):
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            self.cluster_dict[cluster_id] = {
                'initial_centroid': centroid,
                'centroid': centroid,
                'cumulative_residual': self.movement_threshold // 2,
                'cluster': cluster,
                'missed_frames': 0
            }

    def match_clusters(self, current_centroids, current_clusters, max_distance):
        matched_current_indices = set()  # 매칭된 현재 프레임 클러스터의 인덱스
        matched_existing_cluster_ids = set()  # 매칭된 기존 클러스터 ID
        temporary_matches = {}  # 기존 클러스터 ID에 대해 가장 짧은 거리로 매칭된 현재 클러스터 정보 저장

        # 1. 모든 현재 Centriod에 대해 KD-Tree로 KNN 수행
        for current_index, current_centroid in enumerate(current_centroids):
            # KD-Tree로 현재 Centriod에 가장 가까운 가존 Centriod 탐색
            distance, existing_index = self.existing_tree.query(current_centroid, distance_upper_bound=max_distance)

            if distance != np.inf and existing_index < len(self.cluster_dict):
                existing_cluster_id = list(self.cluster_dict.keys())[existing_index]

                # 기존 클러스터 ID에 대해 더 짧은 거리의 매칭이 발견된 경우 갱신
                if (
                    existing_cluster_id not in temporary_matches
                    or distance < temporary_matches[existing_cluster_id]['distance']
                ):
                    temporary_matches[existing_cluster_id] = {
                        'distance': distance,
                        'current_index': current_index,
                        'current_centroid': current_centroid,
                    }

        # 2. 최적의 매칭을 기준으로 클러스터 갱신
        for existing_cluster_id, match_info in temporary_matches.items():
            current_index = match_info['current_index']
            current_centroid = match_info['current_centroid']
            movement_distance = np.linalg.norm(
                current_centroid - self.cluster_dict[existing_cluster_id]['centroid']
            )

            movement_distance = 0 if movement_distance < self.movement_threshold * 0.05 else movement_distance

            # 클러스터 정보 업데이트
            self.cluster_dict[existing_cluster_id]['cumulative_residual'] = (
                self.cluster_dict[existing_cluster_id]['cumulative_residual'] * self.decay_rate + movement_distance
            )
            self.cluster_dict[existing_cluster_id]['centroid'] = current_centroid
            self.cluster_dict[existing_cluster_id]['cluster'] = current_clusters[current_index]
            self.cluster_dict[existing_cluster_id]['missed_frames'] = 0

            matched_current_indices.add(current_index)
            matched_existing_cluster_ids.add(existing_cluster_id)

        return matched_current_indices, matched_existing_cluster_ids



    def detect(self, current_frame):
        clusters, centroids, labels = self.cluster_frame(current_frame)
        self.match_and_update_clusters(clusters, centroids)

        moving_clusters = []
        residual_list = []

        for cluster_id, cluster_info in self.cluster_dict.items():
            cumulative_residual = cluster_info['cumulative_residual']
            missed_frames = cluster_info['missed_frames']
            displacement = np.linalg.norm(cluster_info['centroid'] - cluster_info['initial_centroid'])

            if (cumulative_residual > self.movement_threshold and displacement > self.displacement_threshold) and missed_frames == 0:
                moving_clusters.append(cluster_info['cluster'])
                residual_list.append({
                    'cluster_id': cluster_id,
                    'cumulative_residual': cumulative_residual,
                    'displacement': displacement
                })

        return moving_clusters, residual_list
