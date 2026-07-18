#!/usr/bin/env python3
"""
Nova Carter Circular Path Data Collection Script

This script starts Isaac Sim, loads the Data_Collection_World.usda, and runs 
the Nova Carter robot in a circular path (3m radius) around the boxes while 
collecting camera data (RGB, depth, camera parameters) from all cameras,
instance segmentation from Nova Carter cameras, and LiDAR point cloud data.
"""

import os
import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Isaac Sim standalone application setup
from isaacsim import SimulationApp

# Configure simulation settings for headless remote server
CONFIG = {
    "headless": True,  # Headless mode for remote server
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
    "multi_gpu": False,  # Disable multi-GPU to avoid CUDA errors
    "physics_gpu": 0,    # Use GPU 0 for physics
}

simulation_app = SimulationApp(CONFIG)

# Import Isaac Sim modules after SimulationApp is created
import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.robots import Robot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom, Usd

# Replicator imports for synthetic data
import omni.replicator.core as rep


class NovaCarterCircleDataCollector:
    """Manages Nova Carter robot circular motion and data collection."""
    
    def __init__(self, world_usd_path: str, output_dir: str):
        """
        Initialize the data collector.
        
        Args:
            world_usd_path: Path to the Data_Collection_World.usda file
            output_dir: Directory to save collected data
        """
        self.world_usd_path = world_usd_path
        self.output_dir = output_dir
        self.world = None
        self.robot = None
        self.controller = None
        self.cameras = {}
        self.lidar = None
        self.video_writers = {}  # Dictionary to store OpenCV video writers
        
        # Initialize Colormaps
        # Vibrant colormap for segmentation (gnuplot2)
        try:
            cmap = plt.get_cmap('gnuplot2')
            # Create a lookup table (256, 1, 3) for OpenCV
            self.seg_colormap = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            self.seg_colormap = self.seg_colormap[:, ::-1]
        except Exception:
            # Fallback to OpenCV colormap if matplotlib fails
            self.seg_colormap = None 
        
        # Circular motion parameters
        self.circle_radius = 5.0  # meters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = self.linear_speed / self.circle_radius  # rad/s (negative for clockwise)
        
        # Nova Carter wheel parameters (approximate)
        self.wheel_radius = 0.082  # meters
        self.wheel_base = 0.56     # meters (track width)
        
        # Data collection settings
        self.frame_count = 0
        self.save_interval = 30  # Save every N frames
        self.max_frames =  12*60*30   # Total frames to collect (about 12 minute at 30fps)
        
        # Camera paths
        self.global_camera_paths = [
            "/World/GlobalCameras/Camera_01",
            "/World/GlobalCameras/Camera_02",
            "/World/GlobalCameras/Camera_03",
            "/World/GlobalCameras/Camera_04",
            "/World/GlobalCameras/Camera_05",
            "/World/GlobalCameras/Camera_06",
            "/World/GlobalCameras/Camera_07",
            "/World/GlobalCameras/Camera_08",
            "/World/GlobalCameras/Camera_09",
            "/World/GlobalCameras/Camera_10",
            "/World/GlobalCameras/Camera_11",
            "/World/GlobalCameras/Camera_12",
            "/World/GlobalCameras/Camera_13",
            "/World/GlobalCameras/Camera_14",
            "/World/GlobalCameras/Camera_15",
            "/World/GlobalCameras/Camera_16",
            "/World/GlobalCameras/Camera_17",
            "/World/GlobalCameras/Camera_18",
            "/World/GlobalCameras/Camera_19",
            "/World/GlobalCameras/Camera_20",
        ]
        
        self.nova_carter_camera_paths = [
            "/World/nova_carter/chassis_link/sensors/left_hawk/left/camera_left",
            "/World/nova_carter/chassis_link/sensors/left_hawk/right/camera_right",
            "/World/nova_carter/chassis_link/sensors/right_owl/camera",
        ]
        
        self.lidar_path = "/World/nova_carter/chassis_link/sensors/XT_32/PandarXT_32_10hz"
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"collection_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
    def setup_world(self):
        """Load the USD stage and initialize the simulation world."""
        print(f"Loading world from: {self.world_usd_path}")
        
        # Open the USD stage
        open_stage(self.world_usd_path)
        
        # Create the World instance
        self.world = World(stage_units_in_meters=1.0)
        
        # Initialize physics
        self.world.initialize_physics()
        
        print("World loaded successfully")
        
    def setup_robot(self):
        """Initialize the Nova Carter robot and differential controller."""
        robot_prim_path = "/World/nova_carter"
        
        print(f"Setting up robot at: {robot_prim_path}")
        
        # Get the robot as a wheeled robot with differential drive
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path=robot_prim_path,
                name="nova_carter",
                wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            )
        )
        
        # Create differential controller
        self.controller = DifferentialController(
            name="nova_carter_controller",
            wheel_radius=self.wheel_radius,
            wheel_base=self.wheel_base,
        )
        
        print("Robot and controller initialized")
        
    def setup_cameras(self):
        """Initialize all cameras with required annotators."""
        print("Setting up cameras...")
        
        # Setup global cameras (RGB + depth + parameters)
        for cam_path in self.global_camera_paths:
            cam_name = cam_path.split("/")[-1]
            try:
                camera = Camera(
                    prim_path=cam_path,
                    name=f"global_{cam_name}",
                    resolution=(1280, 720),
                )
                camera.initialize()
                camera.add_distance_to_camera_to_frame()
                self.cameras[f"global_{cam_name}"] = {
                    "camera": camera,
                    "type": "global",
                    "path": cam_path,
                }
                print(f"  Added global camera: {cam_name}")
            except Exception as e:
                print(f"  Warning: Could not initialize camera {cam_path}: {e}")
        
        # Setup Nova Carter cameras (RGB + depth + parameters + instance segmentation)
        for cam_path in self.nova_carter_camera_paths:
            cam_name = cam_path.split("/")[-1]
            try:
                camera = Camera(
                    prim_path=cam_path,
                    name=f"nova_{cam_name}",
                    resolution=(1920, 1200),
                )
                camera.initialize()
                camera.add_distance_to_camera_to_frame()
                camera.add_instance_id_segmentation_to_frame()
                self.cameras[f"nova_{cam_name}"] = {
                    "camera": camera,
                    "type": "nova_carter",
                    "path": cam_path,
                }
                print(f"  Added Nova Carter camera: {cam_name}")
            except Exception as e:
                print(f"  Warning: Could not initialize camera {cam_path}: {e}")
        
        print(f"Initialized {len(self.cameras)} cameras")
        
    def setup_lidar(self):
        """Initialize the LiDAR sensor (disabled due to multi-GPU CUDA issues)."""
        # LiDAR disabled - causes CUDA errors on multi-GPU systems
        # To enable, uncomment below and import LidarRtx from omni.isaac.sensor
        print("LiDAR collection disabled (CUDA compatibility)")
        self.lidar = None
        
        # Uncomment below to enable LiDAR:
        # try:
        #     from omni.isaac.sensor import LidarRtx
        #     self.lidar = LidarRtx(
        #         prim_path=self.lidar_path,
        #         name="nova_carter_lidar",
        #     )
        #     self.lidar.initialize()
        #     print("LiDAR initialized")
        # except Exception as e:
        #     print(f"Warning: Could not initialize LiDAR: {e}")
        #     self.lidar = None
            
    def get_camera_parameters(self, camera_info: dict) -> dict:
        """
        Get intrinsic and extrinsic parameters for a camera.
        
        Returns:
            Dictionary with camera parameters
        """
        camera = camera_info["camera"]
        cam_path = camera_info["path"]
        
        # Get the camera prim
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(cam_path)
        
        params = {
            "path": cam_path,
            "resolution": list(camera.get_resolution()),
        }
        
        # Get intrinsics
        try:
            intrinsics = camera.get_intrinsics_matrix()
            params["intrinsics"] = intrinsics.tolist() if isinstance(intrinsics, np.ndarray) else intrinsics
        except Exception as e:
            params["intrinsics"] = None
            print(f"Warning: Could not get intrinsics for {cam_path}: {e}")
        
        # Get extrinsics (world pose)
        try:
            world_pose = camera.get_world_pose()
            params["extrinsics"] = {
                "position": world_pose[0].tolist() if isinstance(world_pose[0], np.ndarray) else list(world_pose[0]),
                "orientation": world_pose[1].tolist() if isinstance(world_pose[1], np.ndarray) else list(world_pose[1]),
            }
        except Exception as e:
            params["extrinsics"] = None
            print(f"Warning: Could not get extrinsics for {cam_path}: {e}")
            
        return params
        
    def _write_video_frame(self, camera_name: str, stream_name: str, frame_data: np.ndarray):
        """Helper to write a frame to a video file."""
        writer_key = f"{camera_name}_{stream_name}"
        if writer_key not in self.video_writers:
            video_path = os.path.join(self.session_dir, f"{writer_key}.mp4")
            height, width = frame_data.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use 60 FPS as requested
            self.video_writers[writer_key] = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            print(f"Initialized video writer for {writer_key}: {video_path}")
        
        self.video_writers[writer_key].write(frame_data)

    def save_camera_data(self, camera_name: str, camera_info: dict, frame_id: int):
        """Save all data from a camera for the current frame."""
        camera = camera_info["camera"]
        cam_type = camera_info["type"]
        
        # Create camera output directory
        cam_dir = os.path.join(self.session_dir, camera_name)
        os.makedirs(cam_dir, exist_ok=True)
        
        # Get current frame data
        frame = camera.get_current_frame()
        
        # --- Process RGB ---
        rgb_data = None
        if "rgb" in frame:
            rgb_data = frame["rgb"]
        elif "rgba" in frame:
            rgb_data = frame["rgba"]

        if rgb_data is not None:
            # Ensure proper format
            if hasattr(rgb_data, "cpu"): rgb_data = rgb_data.cpu().numpy()
            elif not isinstance(rgb_data, np.ndarray): rgb_data = np.array(rgb_data)
            
            # Remove alpha
            if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 4:
                rgb_data = rgb_data[:, :, :3]
            
            # Ensure uint8
            if rgb_data.dtype != np.uint8:
                if rgb_data.max() <= 1.0: rgb_data = (rgb_data * 255).astype(np.uint8)
                else: rgb_data = rgb_data.astype(np.uint8)

            # 1. Save Raw
            rgb_path = os.path.join(cam_dir, f"rgb_{frame_id:06d}.png")
            try:
                from PIL import Image
                Image.fromarray(rgb_data).save(rgb_path)
            except Exception:
                np.save(rgb_path.replace(".png", ".npy"), rgb_data)
            
            # 2. Write to Video
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            self._write_video_frame(camera_name, "rgb", bgr_data)
        
        # --- Process Depth ---
        depth_data = None
        if "distance_to_camera" in frame:
            depth_data = frame["distance_to_camera"]
        elif "distance_to_image_plane" in frame:
            depth_data = frame["distance_to_image_plane"]
            
        if depth_data is not None:
             # Ensure numpy
            if hasattr(depth_data, "cpu"): depth_data = depth_data.cpu().numpy()
            elif not isinstance(depth_data, np.ndarray): depth_data = np.array(depth_data)

            # 1. Save Raw
            depth_path = os.path.join(cam_dir, f"depth_{frame_id:06d}.npy")
            np.save(depth_path, depth_data)
            
            # 2. Write to Video
            # Normalize to 0-255 uint8 for video (Grayscale)
            valid_mask = np.isfinite(depth_data)
            if valid_mask.any():
                # Clip deep values to improve contrast (e.g. 10m)
                depth_vis = depth_data.copy()
                depth_vis[~valid_mask] = 0
                depth_vis = np.clip(depth_vis, 0, 50.0) # Clip to 10m for better viz
                depth_vis = (depth_vis / 50.0 * 255).astype(np.uint8)
                
                # Make 3-channel BGR for video writer (although it's grayscale)
                depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                self._write_video_frame(camera_name, "depth", depth_bgr)

        # --- Process Instance Segmentation (Nova Carter only) ---
        if cam_type == "nova_carter" and "instance_id_segmentation" in frame:
            seg_raw = frame["instance_id_segmentation"]
            
            # The segmentation data may be a dict with 'data' key or similar structure
            if isinstance(seg_raw, dict):
                # Try common key names
                if "data" in seg_raw:
                    seg_data = seg_raw["data"]
                elif "info" in seg_raw and "data" in seg_raw.get("info", {}):
                    seg_data = seg_raw["info"]["data"]
                else:
                    # Debug: print keys to understand structure
                    print(f"[DEBUG] Seg data is dict with keys: {list(seg_raw.keys())}")
                    seg_data = None
            else:
                seg_data = seg_raw
            
            if seg_data is not None:
                if hasattr(seg_data, "cpu"): seg_data = seg_data.cpu().numpy()
                elif not isinstance(seg_data, np.ndarray): seg_data = np.array(seg_data)

                # 1. Save Raw
                seg_path = os.path.join(cam_dir, f"instance_seg_{frame_id:06d}.npy")
                np.save(seg_path, seg_data)
                
                # Save mapping if available
                if "instance_id_segmentation_mapping" in frame:
                    with open(os.path.join(cam_dir, f"instance_mapping_{frame_id:06d}.json"), "w") as f:
                        json.dump(frame["instance_id_segmentation_mapping"], f, indent=2)

                # 2. Write to Video
                # Normalize IDs to 0-255 for colormap application
                seg_norm = (seg_data % 255).astype(np.uint8)
                
                if self.seg_colormap is not None:
                    try:
                        seg_vis = self.seg_colormap[seg_norm]
                        if len(seg_vis.shape) == 3:
                            self._write_video_frame(camera_name, "seg", seg_vis)
                    except Exception as e:
                        print(f"Error applying seg colormap: {e}")
                else:
                    seg_vis = cv2.applyColorMap(seg_norm, cv2.COLORMAP_JET)
                    self._write_video_frame(camera_name, "seg", seg_vis)

        # Save camera parameters
        # Save camera parameters (Global: once, Robot: every frame)
        if cam_type == "nova_carter" or frame_id == 0:
            params = self.get_camera_parameters(camera_info)
            params["frame_id"] = frame_id
            params_path = os.path.join(cam_dir, f"params_{frame_id:06d}.json")
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2)
            
            # Save instance ID to label mapping if available
            if "instance_id_segmentation_mapping" in frame:
                mapping_path = os.path.join(cam_dir, f"instance_mapping_{frame_id:06d}.json")
                with open(mapping_path, "w") as f:
                    json.dump(frame["instance_id_segmentation_mapping"], f, indent=2)
                    
    def save_lidar_data(self, frame_id: int):
        """Save LiDAR point cloud data."""
        if self.lidar is None:
            return
            
        lidar_dir = os.path.join(self.session_dir, "lidar")
        os.makedirs(lidar_dir, exist_ok=True)
        
        try:
            # Get point cloud data
            point_cloud = self.lidar.get_current_frame()
            
            if point_cloud is not None and "point_cloud" in point_cloud:
                pc_path = os.path.join(lidar_dir, f"pointcloud_{frame_id:06d}.npy")
                np.save(pc_path, point_cloud["point_cloud"])
                
                # Save additional LiDAR data if available
                if "intensities" in point_cloud:
                    intensity_path = os.path.join(lidar_dir, f"intensity_{frame_id:06d}.npy")
                    np.save(intensity_path, point_cloud["intensities"])
                    
        except Exception as e:
            print(f"Warning: Could not save LiDAR data for frame {frame_id}: {e}")
            
    def apply_circular_motion(self):
        """Apply wheel commands to move the robot in a circle."""
        if self.robot is None or self.controller is None:
            return
            
        # Command for circular motion: [linear_velocity, angular_velocity]
        command = [self.linear_speed, self.angular_speed]
        
        # Get wheel action from differential controller
        # Returns an ArticulationAction object
        wheel_action = self.controller.forward(command)
        
        # Apply wheel actions to the robot
        try:
            self.robot.apply_wheel_actions(wheel_action)
        except Exception as e:
            print(f"Warning: Could not apply wheel actions: {e}")
            
    def collect_data_step(self):
        """Collect data for the current frame if interval is met."""
        if self.frame_count % self.save_interval == 0:
            frame_id = self.frame_count // self.save_interval
            
            # Save data from all cameras
            for cam_name, cam_info in self.cameras.items():
                self.save_camera_data(cam_name, cam_info, frame_id)
            
            # Save LiDAR data
            self.save_lidar_data(frame_id)
            
            if frame_id % 10 == 0:
                print(f"Saved frame {frame_id} (simulation step {self.frame_count})")
                
    def run(self):
        """Main execution loop."""
        print("\n" + "="*60)
        print("Nova Carter Circular Path Data Collection")
        print("="*60 + "\n")
        
        # Setup
        self.setup_world()
        self.setup_robot()
        self.setup_cameras()
        self.setup_lidar()
        
        # Reset world
        self.world.reset()

        # Introspection (Debug)
        if self.cameras:
             first_cam = list(self.cameras.values())[0]["camera"]
             print(f"Camera methods: {[m for m in dir(first_cam) if 'add_' in m]}")

        print("Warming up simulation (60 steps)...")
        for i in range(2):
             self.world.step(render=True)
        print("Warmup complete.")
        
        print(f"\nStarting data collection...")
        print(f"  Output directory: {self.session_dir}")
        print(f"  Circle radius: {self.circle_radius}m")
        print(f"  Linear speed: {self.linear_speed}m/s")
        print(f"  Angular speed: {self.angular_speed:.3f}rad/s")
        print(f"  Saving every {self.save_interval} frames")
        print(f"  Max frames: {self.max_frames}")
        print("\nPress Ctrl+C to stop early\n")
        
        try:
            while self.frame_count < self.max_frames:
                # Apply circular motion
                self.apply_circular_motion()
                
                # Step simulation
                self.world.step(render=True)
                
                # Collect data
                try:
                    self.collect_data_step()
                except Exception as e:
                    import traceback
                    print(f"Error collecting data at frame {self.frame_count}: {e}")
                    traceback.print_exc()
                
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                     print(f"Step {self.frame_count}/{self.max_frames}", flush=True)
                
        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user")
            
        print(f"\nData collection complete!")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Saved samples: {self.frame_count // self.save_interval}")
        print(f"  Output: {self.session_dir}")
        
    def cleanup(self):
        """Clean up resources."""
        # Release all video writers
        for name, writer in self.video_writers.items():
            print(f"Releasing video writer for {name}")
            writer.release()
            
        if self.world:
            self.world.stop()
        simulation_app.close()


def main():
    """Main entry point."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    world_usd_path = os.path.join(script_dir, "Collected_New_warehouse/New_warehouse.usda")
    output_dir = os.path.join(script_dir, "Collected_warehouse")
    
    # Verify world file exists
    if not os.path.exists(world_usd_path):
        print(f"Error: World USD file not found: {world_usd_path}")
        sys.exit(1)
        
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and run collector
    collector = NovaCarterCircleDataCollector(world_usd_path, output_dir)
    
    try:
        collector.run()
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
