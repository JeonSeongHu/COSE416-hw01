import open3d as o3d
import torch
import numpy as np
import os
import matplotlib.pyplot as plt  # For color maps

from utils.preprocessor import PointCloudPreprocessor
from utils.PedestrianDetector import PedestrianDetector

class PCDViewer:
    def __init__(self, window_name="PCD Viewer", axis_size=0.05, window_width=1024, window_height=1024):
        """
        Initialize the PCDViewer class.

        Parameters:
        - window_name (str): Name of the Open3D visualization window.
        - axis_size (float): Size of the coordinate axes.
        - window_width (int): Width of the visualization window.
        - window_height (int): Height of the visualization window.
        """
        self.window_name = window_name
        self.window_width = window_width
        self.window_height = window_height

        # Ensure window dimensions are positive
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError("Window width and height must be greater than zero.")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=self.window_name, width=self.window_width, height=self.window_height)

        self.current_index = 0
        self.pcd_data = []  # Storage for PCD data to visualize
        self.residuals_and_clusters = []  # Storage for residuals and clusters data
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])

        # Initialize camera parameters
        self.camera_params = None

        # Register keyboard callback functions
        self.vis.register_key_callback(ord("N"), self.next_frame)
        self.vis.register_key_callback(ord("P"), self.prev_frame)
        self.vis.register_key_callback(ord("Q"), self.quit_viewer)

    def update_camera_intrinsics(self):
        """
        Ensure that camera intrinsics match the current window size.
        """
        if self.camera_params is not None:
            fx, fy = self.camera_params.intrinsic.get_focal_length()
            # Open3D requires cx and cy to be at the image center
            cx = (self.window_width - 1) / 2.0
            cy = (self.window_height - 1) / 2.0

            # Ensure fx and fy are valid
            fx = fx if fx > 0 else self.window_width / 2.0
            fy = fy if fy > 0 else self.window_height / 2.0

            # Update intrinsic parameters with valid values
            self.camera_params.intrinsic.set_intrinsics(
                width=self.window_width,
                height=self.window_height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy
            )

    def load_frame(self, index):
        self.vis.clear_geometries()
        if 0 <= index < len(self.pcd_data):
            static_pcd = self.pcd_data[index]
            static_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.add_geometry(static_pcd)
            self.vis.add_geometry(self.coordinate_frame)

            residual_list, moving_clusters = self.residuals_and_clusters[index]
            self.add_clusters(moving_clusters)
            self.get_pcd_statistics(self.pcd_data[index])
        else:
            print(f"Invalid index: {index}")
            return

        if self.camera_params is not None:
            try:
                view_control = self.vis.get_view_control()
                self.update_camera_intrinsics()  # Ensure intrinsic parameters match the window
                view_control.convert_from_pinhole_camera_parameters(self.camera_params)
            except Exception as e:
                print(f"Warning: Failed to apply camera pose. {e}")
        else:
            self.vis.poll_events()
            self.vis.update_renderer()
            self.get_camera_pose()

        self.vis.poll_events()
        self.vis.update_renderer()
        print(f"Loaded frame {index + 1}/{len(self.pcd_data)}.")

    def add_clusters(self, moving_clusters):
        """
        Add clusters and assign unique colors to each for better visualization.

        Parameters:
        - moving_clusters (list of np.ndarray): List of cluster points.
        """
        if not moving_clusters:
            print("No clusters to visualize.")
            return

        # Generate unique colors for each cluster
        num_clusters = len(moving_clusters)
        colors = self.get_n_colors(num_clusters)

        for idx, cluster in enumerate(moving_clusters):
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)

            # Assign a unique color to each cluster
            color = colors[idx]
            cluster_pcd.paint_uniform_color(color)
            self.vis.add_geometry(cluster_pcd)

            # Add bounding box
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = color
            self.vis.add_geometry(bbox)

    def get_n_colors(self, n):
        """
        Generate n unique colors.

        Parameters:
        - n (int): Number of colors to generate.

        Returns:
        - list of RGB colors.
        """
        colors = plt.cm.get_cmap('hsv', n)
        return [colors(i)[:3] for i in range(n)]

    def get_pcd_statistics(self, pcd):
        """
        Print basic statistics of the given point cloud.

        Parameters:
        - pcd (o3d.geometry.PointCloud): Point cloud.
        """
        points = np.asarray(pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = points.mean(axis=0)

        print(f"Point Cloud Statistics:")
        print(f"  Min Bound: {min_bound}")
        print(f"  Max Bound: {max_bound}")
        print(f"  Center: {center}")

    def get_camera_pose(self):
        """
        Get and store the current camera parameters of the visualization window.
        """
        view_control = self.vis.get_view_control()
        self.camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Ensure intrinsic parameters match the window size
        fx, fy = self.camera_params.intrinsic.get_focal_length()
        # Open3D requires cx and cy to be at the image center
        cx = (self.window_width - 1) / 2.0
        cy = (self.window_height - 1) / 2.0

        # Ensure fx and fy are valid
        fx = fx if fx > 0 else self.window_width / 2.0
        fy = fy if fy > 0 else self.window_height / 2.0

        self.camera_params.intrinsic.set_intrinsics(
            width=self.window_width,
            height=self.window_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

    def next_frame(self, vis):
        """
        Callback function to load the next frame.

        Parameters:
        - vis (open3d.visualization.Visualizer): Open3D visualization object.
        """
        if self.current_index < len(self.pcd_data) - 1:
            self.get_camera_pose()  # Store current camera pose
            self.current_index += 1
            self.load_frame(self.current_index)
        else:
            print("Already at the last frame.")
        return False

    def prev_frame(self, vis):
        """
        Callback function to load the previous frame.

        Parameters:
        - vis (open3d.visualization.Visualizer): Open3D visualization object.
        """
        if self.current_index > 0:
            self.get_camera_pose()  # Store current camera pose
            self.current_index -= 1
            self.load_frame(self.current_index)
        else:
            print("Already at the first frame.")
        return False

    def quit_viewer(self, vis):
        """
        Callback function to exit the visualization window.

        Parameters:
        - vis (open3d.visualization.Visualizer): Open3D visualization object.
        """
        print("Exiting viewer.")
        self.vis.destroy_window()
        return False

    def run(self, processed_sequence, residuals_and_clusters):
        """
        Run the visualization window.

        Parameters:
        - processed_sequence (list of torch.Tensor): Sequence of point clouds.
        - residuals_and_clusters (list of tuples): (residual_list, moving_clusters) pairs for each frame.
        """
        if not processed_sequence:
            print("No PCD data loaded. Please load PCD data before running the viewer.")
            return

        self.pcd_data = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame.cpu().numpy()))
            for frame in processed_sequence
        ]
        self.residuals_and_clusters = residuals_and_clusters

        self.load_frame(self.current_index)
        self.vis.run()
        self.vis.destroy_window()

def main():
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 전처리기 초기화
    preprocessor = PointCloudPreprocessor(device=device, voxel_size=0.2)

    # 보행자 검출기 초기화
    detector = PedestrianDetector(eps=0.01, min_samples=7, movement_threshold=0.01, 
                                  decay_rate=0.9, displacement_threshold=0.04, device=device)

    # 데이터 폴더 경로
    folder_path = "data/01_straight_walk/pcd"

    # 포인트 클라우드 시퀀스 전처리
    processed_sequence = preprocessor.process_folder(folder_path, num=100)
    print(f"Processed {len(processed_sequence)} frames from folder.")

    # Residuals and clusters collection
    residuals_and_clusters = []
    for frame in processed_sequence:
        moving_clusters, residual_list = detector.detect(frame)
        residuals_and_clusters.append((residual_list, moving_clusters))

    # PCD Viewer 실행
    viewer = PCDViewer(window_name="Pedestrian Detection Viewer", axis_size=0.05)
    viewer.run(processed_sequence, residuals_and_clusters)

if __name__ == "__main__":
    main()
