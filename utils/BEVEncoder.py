import torch
import numpy as np

class BEVEncoder:
    def __init__(self, x_range, y_range, z_range, voxel_size):
        """
        Initialize BEV Encoder.

        Parameters:
        - x_range (tuple): (min_x, max_x) in meters.
        - y_range (tuple): (min_y, max_y) in meters.
        - z_range (tuple): (min_z, max_z) in meters.
        - voxel_size (float): Size of each voxel in meters.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.voxel_size = voxel_size

        self.grid_size = (
            int((x_range[1] - x_range[0]) / voxel_size),
            int((y_range[1] - y_range[0]) / voxel_size)
        )

    def encode(self, point_cloud):
        """
        Encode a single point cloud into BEV.

        Parameters:
        - point_cloud (torch.Tensor): (N, 3) tensor of points (x, y, z).

        Returns:
        - bev_map (torch.Tensor): (H, W) tensor representing BEV.
        """
        # Filter points within range
        mask = (
            (point_cloud[:, 0] >= self.x_range[0]) & (point_cloud[:, 0] <= self.x_range[1]) &
            (point_cloud[:, 1] >= self.y_range[0]) & (point_cloud[:, 1] <= self.y_range[1]) &
            (point_cloud[:, 2] >= self.z_range[0]) & (point_cloud[:, 2] <= self.z_range[1])
        )
        filtered_points = point_cloud[mask]

        # Compute voxel indices
        indices = ((filtered_points[:, :2] - torch.tensor([self.x_range[0], self.y_range[0]], device=filtered_points.device)) 
                   / self.voxel_size).long()

        # Create empty BEV map
        bev_map = torch.zeros(self.grid_size, device=filtered_points.device)

        # Populate BEV map with point density
        for idx in indices:
            bev_map[idx[1], idx[0]] += 1  # Note: y corresponds to rows, x to columns

        return bev_map

    def encode_sequence(self, point_cloud_sequence):
        """
        Encode a sequence of point clouds into BEV maps.

        Parameters:
        - point_cloud_sequence (list of torch.Tensor): List of (N_i, 3) tensors.

        Returns:
        - bev_sequence (torch.Tensor): (B, H, W) tensor of BEV maps.
        """
        bev_sequence = []
        for point_cloud in point_cloud_sequence:
            bev_map = self.encode(point_cloud)
            bev_sequence.append(bev_map)

        return torch.stack(bev_sequence)


# test code
if __name__ == '__main__':
    # Define point cloud range and voxel size
    x_range = (-1, 1)  # meters
    y_range = (-1, 1)  # meters
    z_range = (-1, 1)    # meters
    voxel_size = 0.1     # meters

    # Initialize BEV Encoder
    bev_encoder = BEVEncoder(x_range, y_range, z_range, voxel_size)

    # Example preprocessed list of tensors (list of (N, 3) torch.Tensor)
    preprocessed_sequence = [torch.randn(1000, 3), torch.randn(1200, 3), torch.randn(1100, 3)]

    # Encode single frame
    single_bev = bev_encoder.encode(preprocessed_sequence[0])
    print("Single BEV map shape:", single_bev.shape)

    # Encode entire sequence
    bev_sequence = bev_encoder.encode_sequence(preprocessed_sequence)
    print("BEV sequence shape:", bev_sequence.shape)
