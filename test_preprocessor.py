import open3d as o3d
import torch
import numpy as np
from utils.preprocessor import PointCloudPreprocessor
from utils.BEVEncoder import BEVEncoder
from viewer.PCDViewer import PCDViewer

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

import cv2
import os
import matplotlib.pyplot as plt

def create_bev_video(bev_sequence, video_path, fps=5):
    """
    Create a BEV video from a sequence of BEV maps.

    Parameters:
    - bev_sequence (torch.Tensor): (B, H, W) tensor of BEV maps.
    - video_path (str): Path to save the output video.
    - fps (int): Frames per second for the video.
    """
    bev_height, bev_width = bev_sequence[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (bev_width, bev_height))

    # Write each BEV frame to the video
    for i, bev_map in enumerate(bev_sequence):
        frame = bev_to_frame(bev_map.cpu().numpy())
        video_writer.write(frame)
        print(f"Added frame {i + 1}/{len(bev_sequence)} to BEV video.")

    # Release the video writer
    video_writer.release()
    print(f"BEV video saved to: {video_path}")


def create_residual_bev_video(processed_sequence, video_path, voxel_size=0.1, fps=5):
    """
    Create a residual BEV video from a sequence of point clouds.

    Parameters:
    - processed_sequence (list of torch.Tensor): Preprocessed point cloud sequence.
    - video_path (str): Path to save the output video.
    - voxel_size (float): Size of each voxel in meters.
    - fps (int): Frames per second for the video.
    """
    # Define BEV Encoder parameters
    x_range = (-0.5, 0.5)  # meters
    y_range = (-0.5, 0.5)  # meters
    z_range = (-0.5, 0.5)    # meters

    # Initialize BEV Encoder
    bev_encoder = BEVEncoder(x_range, y_range, z_range, voxel_size)

    # Encode the sequence into BEV maps
    bev_sequence = bev_encoder.encode_sequence(processed_sequence)
    print(f"BEV sequence shape: {bev_sequence.shape}")  # Shape: (B, H, W)

    # Save the full BEV video
    full_bev_video_path = video_path.replace("residual_", "full_")
    create_bev_video(bev_sequence, full_bev_video_path, fps=fps)

    # Compute residual BEV maps
    residual_bevs = []
    for i in range(1, len(bev_sequence)):
        residual_bev = bev_sequence[i] - bev_sequence[i - 1]
        residual_bevs.append(residual_bev.cpu().numpy())

    # Create video writer for residual BEV maps
    bev_height, bev_width = residual_bevs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (bev_width, bev_height))

    # Write each residual BEV frame to the video
    for i, residual in enumerate(residual_bevs):
        frame = residual_to_frame(residual)
        video_writer.write(frame)
        print(f"Added frame {i + 1}/{len(residual_bevs)} to residual BEV video.")

    # Release the video writer
    video_writer.release()
    print(f"Residual BEV video saved to: {video_path}")


def bev_to_frame(bev_map):
    """
    Convert a BEV map to a frame suitable for video writing.

    Parameters:
    - bev_map (np.ndarray): 2D BEV map.

    Returns:
    - frame (np.ndarray): 3-channel (BGR) frame for video writing.
    """
    # Normalize BEV values for visualization
    bev_normalized = (bev_map - bev_map.min()) / (bev_map.max() - bev_map.min())
    bev_normalized = (bev_normalized * 255).astype(np.uint8)

    # Convert to a 3-channel BGR image for video writing
    frame = cv2.applyColorMap(bev_normalized, cv2.COLORMAP_HOT)
    return frame


def residual_to_frame(residual_bev):
    """
    Convert a residual BEV map to a frame suitable for video writing.

    Parameters:
    - residual_bev (np.ndarray): 2D residual BEV map.

    Returns:
    - frame (np.ndarray): 3-channel (BGR) frame for video writing.
    """
    # Normalize residual values for visualization
    residual_normalized = (residual_bev - residual_bev.min()) / (residual_bev.max() - residual_bev.min())
    residual_normalized = (residual_normalized * 255).astype(np.uint8)

    # Convert to a 3-channel BGR image for video writing
    frame = cv2.applyColorMap(residual_normalized, cv2.COLORMAP_HOT)
    return frame


if __name__ == "__main__":
    # Preprocess the point cloud sequence
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = PointCloudPreprocessor(device=device)

    folder_path = "data/01_straight_walk/pcd"
    processed_sequence = preprocessor.process_folder(folder_path, num=30)
    print(f"Processed {len(processed_sequence)} frames from folder.")

    # Create and save BEV and Residual BEV videos
    residual_video_path = "output/residual_bev.mp4"
    os.makedirs(os.path.dirname(residual_video_path), exist_ok=True)
    create_residual_bev_video(processed_sequence, residual_video_path, voxel_size=0.0005, fps=5)
