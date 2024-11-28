import open3d as o3d
import torch
import numpy as np
import os
import matplotlib.pyplot as plt  # For color maps

from utils.preprocessor import PointCloudPreprocessor
from utils.PedestrianDetector import PedestrianDetector
from viewer.PCDViewer import PCDViewer
from tqdm import tqdm

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = PointCloudPreprocessor(
        device=device, 
        voxel_size=0.3,
        sor_neighbors=10,
        sor_std_ratio=1.0,
        ror_nb_points=6,
        ror_radius=0.6,
        plane_distance_threshold=0.2,
        plane_num_iterations=5000,
        apply_sor=False,
        apply_ror=True,
        apply_plane_removal=True,
        ignore_preprocessing=False,
    )

    detector = PedestrianDetector(
        eps=0.007, 
        min_samples=4, 
        max_samples=30, 
        movement_threshold=0.01,
        decay_rate=0.9, 
        displacement_threshold=0.03, 
        device=device,
        max_cluster_size=0.1,
        missed_frames_threshold=3,
    )

    FOLDERS = [
        # "data/01_straight_walk/pcd",
        # "data/02_straight_duck_walk/pcd",
        # "data/03_straight_crawl/pcd",
        # "data/04_zigzag_walk/pcd",
        # "data/05_straight_duck_walk/pcd",
        "data/06_straight_crawl/pcd",
        # "data/07_straight_walk/pcd",
    ]

    # FOLDERS = [FOLDERS[4]]  # For testing, only process the first folder
    output_video_dir = "output_videos"
    os.makedirs(output_video_dir, exist_ok=True)

    for folder_index, folder_path in enumerate(FOLDERS):
        print(f"Processing folder: {folder_path}")
        output_video_path = os.path.join(output_video_dir, f"output_{folder_path.split('/')[1]}.mp4")

        processed_sequence = preprocessor.process_folder(folder_path, start_frame=0, end_frame=200)
        print(f"Processed {len(processed_sequence)} frames from {folder_path}.")

        residuals_and_clusters = []
        for frame in tqdm(processed_sequence, desc=f"Detecting in folder {folder_index+1}/{len(FOLDERS)}"):
            moving_clusters, residual_list = detector.detect(frame)
            residuals_and_clusters.append((residual_list, moving_clusters))

        viewer = PCDViewer(window_name=f"Viewer for {os.path.basename(folder_path)}", axis_size=0.05)
        # viewer.run(processed_sequence, residuals_and_clusters)
        viewer.save_to_video(processed_sequence, residuals_and_clusters, output_path=output_video_path, fps=20)
        # print(f"Saved video for {folder_path} to {output_video_path}.")

        del viewer

if __name__ == "__main__":
    main()
