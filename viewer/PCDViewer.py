import open3d as o3d
import numpy as np
import torch
import os

class PCDViewer:
    def __init__(self, window_name="PCD Viewer", axis_size=0.1):
        """
        PCDViewer 클래스 초기화.

        Parameters:
        - window_name (str): Open3D 시각화 창의 이름.
        - axis_size (float): 좌표축의 크기.
        """
        self.window_name = window_name
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=self.window_name)
        self.current_index = 0
        self.pcd_data = []  # 시각화할 PCD 데이터 저장용
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])

        # 키보드 콜백 함수 등록
        self.vis.register_key_callback(ord("N"), self.next_frame)
        self.vis.register_key_callback(ord("P"), self.prev_frame)
        self.vis.register_key_callback(ord("Q"), self.quit_viewer)

    def load_pcd_from_folder(self, folder_path):
        """
        폴더에서 PCD 파일 시퀀스를 로드.

        Parameters:
        - folder_path (str): PCD 파일들이 저장된 폴더의 경로.
        """
        pcd_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(".pcd")
        ])  # 파일 이름만 정렬

        self.pcd_data = [
            o3d.io.read_point_cloud(os.path.join(folder_path, f))
            for f in pcd_files
        ]  # 파일을 읽어 PointCloud로 변환

        print(f"Loaded {len(self.pcd_data)} PCD files from folder: {folder_path}")
        self.current_index = 0

    def load_pcd_from_file(self, file_path):
        """
        단일 PCD 파일을 로드.

        Parameters:
        - file_path (str): PCD 파일 경로.
        """
        pcd = o3d.io.read_point_cloud(file_path)
        self.pcd_data = [pcd]  # 리스트 형태로 저장
        print(f"Loaded single PCD file: {file_path}")
        self.current_index = 0

    def load_pcd_from_array(self, array):
        """
        numpy 배열 또는 torch.Tensor로부터 PCD 데이터를 로드.

        Parameters:
        - array (np.ndarray or torch.Tensor): (N, 3) 형태의 포인트 클라우드 데이터.
        """
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()
        elif not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray or torch.Tensor.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(array)
        self.pcd_data = [pcd]  # 리스트 형태로 저장
        print("Loaded PCD from array.")
        self.current_index = 0

    def load_pcd_from_array_set(self, arrays):
        """
        여러 numpy 배열 또는 torch.Tensor로부터 PCD 데이터를 로드.

        Parameters:
        - arrays (list of np.ndarray or torch.Tensor): (N, 3) 형태의 포인트 클라우드 데이터들의 리스트.
        """
        self.pcd_data = []
        for i, array in enumerate(arrays):
            if isinstance(array, torch.Tensor):
                array = array.cpu().numpy()
            elif not isinstance(array, np.ndarray):
                raise TypeError("Each element must be a numpy.ndarray or torch.Tensor.")
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(array)
            self.pcd_data.append(pcd)

        print(f"Loaded {len(self.pcd_data)} PCDs from array set.")
        self.current_index = 0

    def load_frame(self, index, camera_pose=None):
        """
        지정된 인덱스의 PCD 데이터를 로드하고 시각화를 갱신.

        Parameters:
        - index (int): 로드할 PCD 데이터의 인덱스.
        - camera_pose (o3d.camera.PinholeCameraParameters, optional): 이전 프레임의 카메라 포즈.
        """
        self.vis.clear_geometries()
        if 0 <= index < len(self.pcd_data):
            self.vis.add_geometry(self.pcd_data[index])
            self.vis.add_geometry(self.coordinate_frame)  # 좌표계 추가
            self.get_pcd_statistics(self.pcd_data[index])  # 통계 출력
        else:
            print(f"Invalid index: {index}")
            return

        if camera_pose:
            try:
                view_control = self.vis.get_view_control()
                view_control.convert_from_pinhole_camera_parameters(camera_pose)
            except Exception as e:
                print(f"Warning: Failed to apply camera pose. {e}")

        print(f"Loaded frame {index + 1}/{len(self.pcd_data)}.")

    def get_pcd_statistics(self, pcd):
        """
        주어진 포인트 클라우드의 대략적인 통계(범위 및 중심)를 출력.

        Parameters:
        - pcd (o3d.geometry.PointCloud): 포인트 클라우드.
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
        현재 시각화 창의 카메라 포즈를 반환.

        Returns:
        - o3d.camera.PinholeCameraParameters: 현재 카메라 포즈.
        """
        view_control = self.vis.get_view_control()
        return view_control.convert_to_pinhole_camera_parameters()

    def next_frame(self, vis):
        """
        다음 프레임을 로드하는 콜백 함수.

        Parameters:
        - vis (open3d.visualization.Visualizer): Open3D 시각화 객체.
        """
        if self.current_index < len(self.pcd_data) - 1:
            camera_pose = self.get_camera_pose()
            self.current_index += 1
            self.load_frame(self.current_index, camera_pose)
        else:
            print("Already at the last frame.")
        return False

    def prev_frame(self, vis):
        """
        이전 프레임을 로드하는 콜백 함수.

        Parameters:
        - vis (open3d.visualization.Visualizer): Open3D 시각화 객체.
        """
        if self.current_index > 0:
            camera_pose = self.get_camera_pose()
            self.current_index -= 1
            self.load_frame(self.current_index, camera_pose)
        else:
            print("Already at the first frame.")
        return False

    def quit_viewer(self, vis):
        """
        시각화 창을 종료하는 콜백 함수.

        Parameters:
        - vis (open3d.visualization.Visualizer): Open3D 시각화 객체.
        """
        print("Exiting viewer.")
        self.vis.destroy_window()
        return False

    def run(self):
        """
        시각화 창을 실행.
        """
        if not self.pcd_data:
            print("No PCD data loaded. Please load PCD data before running the viewer.")
            return

        self.load_frame(self.current_index)
        self.vis.run()
        self.vis.destroy_window()


if __name__ == "__main__":
    viewer = PCDViewer(window_name="Array Viewer", axis_size=0.5)

    # Set of arrays 로드
    viewer.load_pcd_from_folder("data/01_straight_walk/pcd")
    viewer.run()
