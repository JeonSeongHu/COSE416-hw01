import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
 
class PCDViewer:
    def __init__(self, window_name="PCD Viewer", axis_size=0.05, window_width=1024, window_height=1024):
        """
        PCDViewer 클래스를 초기화합니다.

        매개변수:
        - window_name (str): Open3D 시각화 창의 이름.
        - axis_size (float): 좌표축의 크기.
        - window_width (int): 시각화 창의 너비.
        - window_height (int): 시각화 창의 높이.
        """
        self.window_name = window_name
        self.window_width = window_width
        self.window_height = window_height

        # 창의 크기가 양수인지 확인
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError("창의 너비와 높이는 0보다 커야 합니다.")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=self.window_name, width=self.window_width, height=self.window_height)

        self.current_index = 0
        self.pcd_data = []  # 시각화할 PCD 데이터를 저장
        self.residuals_and_clusters = []  # 잔차 및 클러스터 데이터를 저장
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])

        # 초기 카메라 파라미터 설정
        self.initial_camera_parameters = {
            'lookat': np.array([0.0, 0.1, 0.0]),
            'up': np.array([0.0, 0.5, 0.3]),    # 'up' 벡터를 약간 기울였습니다.
            'front': np.array([0, -1.4, 1.0]), # 'front' 벡터를 조정하여 카메라 시점을 변경했습니다.
            'zoom': 0.3
        }

        # 키보드 콜백 함수 등록
        self.vis.register_key_callback(ord("N"), self.next_frame)
        self.vis.register_key_callback(ord("P"), self.prev_frame)
        self.vis.register_key_callback(ord("Q"), self.quit_viewer)

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
            # print(f"유효하지 않은 인덱스입니다: {index}")
            return

        # 초기 카메라 파라미터 적용
        view_control = self.vis.get_view_control()
        view_control.set_lookat(self.initial_camera_parameters['lookat'])
        view_control.set_up(self.initial_camera_parameters['up'])
        view_control.set_front(self.initial_camera_parameters['front'])
        view_control.set_zoom(self.initial_camera_parameters['zoom'])

        self.vis.poll_events()
        self.vis.update_renderer()
        # print(f"{index + 1}/{len(self.pcd_data)} 프레임을 로드했습니다.")

    def add_clusters(self, moving_clusters):
        """
        클러스터를 추가하고 각 클러스터에 고유한 색상을 지정하여 시각화를 개선합니다.

        매개변수:
        - moving_clusters (list of np.ndarray): 클러스터 포인트들의 리스트.
        """
        if not moving_clusters:
            # print("시각화할 클러스터가 없습니다.")
            return

        # 각 클러스터에 대한 고유한 색상 생성
        num_clusters = len(moving_clusters)
        colors = self.get_n_colors(num_clusters)

        for idx, cluster in enumerate(moving_clusters):
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)

            # 각 클러스터에 고유한 색상 지정
            color = colors[idx]
            cluster_pcd.paint_uniform_color(color)
            self.vis.add_geometry(cluster_pcd)

            # 바운딩 박스 추가
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = color
            self.vis.add_geometry(bbox)

    def get_n_colors(self, n):
        """
        n개의 고유한 색상을 생성합니다.

        매개변수:
        - n (int): 생성할 색상의 수.

        반환값:
        - RGB 색상의 리스트.
        """
        colors = plt.cm.get_cmap('hsv', n)
        return [colors(i)[:3] for i in range(n)]

    def get_pcd_statistics(self, pcd):
        """
        주어진 포인트 클라우드의 기본 통계를 출력합니다.

        매개변수:
        - pcd (o3d.geometry.PointCloud): 포인트 클라우드.
        """
        points = np.asarray(pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = points.mean(axis=0)

        # print(f"포인트 클라우드 통계:")
        # print(f"  최소 경계: {min_bound}")
        # print(f"  최대 경계: {max_bound}")
        # print(f"  중심: {center}")

    def next_frame(self, vis):
        """
        다음 프레임을 로드하는 콜백 함수입니다.

        매개변수:
        - vis (open3d.visualization.Visualizer): Open3D 시각화 객체.
        """
        if self.current_index < len(self.pcd_data) - 1:
            self.current_index += 1
            self.load_frame(self.current_index)
        else:
            print("이미 마지막 프레임입니다.")
        return False

    def prev_frame(self, vis):
        """
        이전 프레임을 로드하는 콜백 함수입니다.

        매개변수:
        - vis (open3d.visualization.Visualizer): Open3D 시각화 객체.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.load_frame(self.current_index)
        else:
            print("이미 첫 번째 프레임입니다.")
        return False

    def quit_viewer(self, vis):
        """
        시각화 창을 종료하는 콜백 함수입니다.

        매개변수:
        - vis (open3d.visualization.Visualizer): Open3D 시각화 객체.
        """
        print("뷰어를 종료합니다.")
        self.vis.destroy_window()
        return False

    def run(self, processed_sequence, residuals_and_clusters):
        """
        시각화 창을 실행합니다.

        매개변수:
        - processed_sequence (list of torch.Tensor): 포인트 클라우드 시퀀스.
        - residuals_and_clusters (list of tuples): 각 프레임에 대한 (residual_list, moving_clusters) 쌍.
        """
        if not processed_sequence:
            print("로드된 PCD 데이터가 없습니다. 뷰어를 실행하기 전에 PCD 데이터를 로드하십시오.")
            return

        self.pcd_data = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame.cpu().numpy()))
            for frame in processed_sequence
        ]
        self.residuals_and_clusters = residuals_and_clusters

        self.load_frame(self.current_index)
        self.vis.run()
        self.vis.destroy_window()

    def save_to_video(self, processed_sequence, residuals_and_clusters, output_path="output.mp4", fps=30):
        if not processed_sequence:
            print("PCD 데이터가 없습니다. 비디오를 저장하려면 데이터를 로드하십시오.")
            return

        # 데이터 초기화
        self.pcd_data = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame.cpu().numpy()))
            for frame in processed_sequence
        ]
        self.residuals_and_clusters = residuals_and_clusters

        # 출력 디렉토리가 존재하지 않으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        video_writer = None

        # Open3D 카메라 설정 초기화
        view_control = self.vis.get_view_control()
        view_control.set_lookat(self.initial_camera_parameters['lookat'])
        view_control.set_up(self.initial_camera_parameters['up'])
        view_control.set_front(self.initial_camera_parameters['front'])
        view_control.set_zoom(self.initial_camera_parameters['zoom'])

        for index in range(len(self.pcd_data)):
            self.load_frame(index)
            self.vis.poll_events()
            self.vis.update_renderer()

            # 딜레이 추가 (렌더링 안정성 확보)
            self.vis.poll_events()
            self.vis.update_renderer()

            temp_image_path = f"temp_frame_{index:04d}.png"
            self.vis.capture_screen_image(temp_image_path)
            captured_frame = cv2.imread(temp_image_path)

            if video_writer is None:
                height, width, _ = captured_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            video_writer.write(captured_frame)
            os.remove(temp_image_path)

        if video_writer:
            video_writer.release()

        print(f"비디오가 저장되었습니다: {output_path}")
        self.vis.destroy_window()
