#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Point Cloud Visualizer cho Vision System

Visualize point cloud từ depth map với 3D bounding boxes.

Usage:
    python scripts/visualization/pointcloud_viewer.py \
        --rgb data/01_interim/vision/aligned/rgb/image.png \
        --depth data/01_interim/vision/aligned/depth/image.png \
        --annotation data/02_processed/vision/coco_format/hilo_annotations_3d.json
"""

import argparse
import json
from pathlib import Path
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV không có sẵn")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️  Open3D không có sẵn. Cài đặt: pip install open3d")
    print("   Hoặc dùng PyVista: pip install pyvista")


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Point Cloud Visualizer"
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        help="RGB image path",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        help="Depth image path",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        help="Annotation JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image/video path",
    )
    parser.add_argument(
        "--camera-intrinsics",
        type=Path,
        default=None,
        help="Camera intrinsics file (JSON)",
    )
    return parser.parse_args()


def load_camera_intrinsics(intrinsics_path):
    """Load camera intrinsics."""
    if intrinsics_path and intrinsics_path.exists():
        with intrinsics_path.open() as f:
            data = json.load(f)
            fx = data.get("fx", 525.0)
            fy = data.get("fy", 525.0)
            cx = data.get("cx", 320.0)
            cy = data.get("cy", 240.0)
            return fx, fy, cx, cy
    # Default intrinsics (assume 640x480)
    return 525.0, 525.0, 320.0, 240.0


def depth_to_pointcloud(rgb_img, depth_img, intrinsics=None):
    """Convert depth map to point cloud."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV required")
    
    h, w = depth_img.shape[:2]
    
    if intrinsics is None:
        fx, fy, cx, cy = load_camera_intrinsics(None)
    else:
        fx, fy, cx, cy = intrinsics
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to 3D
    z = depth_img.astype(np.float32) / 1000.0  # Convert mm to m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack coordinates
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # Get colors
    if rgb_img is not None:
        colors = rgb_img.reshape(-1, 3) / 255.0
    else:
        colors = np.ones_like(points)
    
    # Filter invalid points
    valid = z.reshape(-1) > 0
    points = points[valid]
    colors = colors[valid]
    
    return points, colors


def create_bbox_3d(bbox3d):
    """Create 3D bounding box mesh."""
    cx, cy, cz = bbox3d["cx"], bbox3d["cy"], bbox3d["cz"]
    sx, sy, sz = bbox3d["sx"], bbox3d["sy"], bbox3d["sz"]
    
    # 8 corners
    corners = np.array([
        [cx - sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy + sy/2, cz + sz/2],
        [cx - sx/2, cy + sy/2, cz + sz/2],
    ])
    
    # Create box lines
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Front face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Back face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Connecting edges
    ]
    
    return corners, lines


def visualize_with_open3d(points, colors, bboxes=None):
    """Visualize với Open3D."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D required")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Point Cloud Viewer", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Add bounding boxes
    if bboxes:
        for bbox in bboxes:
            corners, lines = create_bbox_3d(bbox["bbox3d"])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
            vis.add_geometry(line_set)
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()


def visualize_with_pyvista(points, colors, bboxes=None):
    """Visualize với PyVista (alternative)."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista required. Install: pip install pyvista")
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add point cloud
    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = (colors * 255).astype(np.uint8)
    plotter.add_mesh(point_cloud, point_size=3, render_points_as_spheres=True)
    
    # Add bounding boxes
    if bboxes:
        for bbox in bboxes:
            corners, lines = create_bbox_3d(bbox["bbox3d"])
            # Create box from corners
            box = pv.Line(corners[lines])
            plotter.add_mesh(box, color="red", line_width=3)
    
    plotter.show()


def main():
    args = parse_args()
    
    print("🔍 3D Point Cloud Visualizer")
    print("="*70)
    
    # Load images
    if not CV2_AVAILABLE:
        print("❌ OpenCV required")
        return
    
    if args.rgb and args.rgb.exists():
        rgb_img = cv2.imread(str(args.rgb))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        print(f"✅ Loaded RGB: {args.rgb}")
    else:
        rgb_img = None
        print("⚠️  No RGB image, using default colors")
    
    if args.depth and args.depth.exists():
        depth_img = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
        print(f"✅ Loaded Depth: {args.depth}")
    else:
        print("❌ Depth image required")
        return
    
    # Load annotations
    bboxes = []
    if args.annotation and args.annotation.exists():
        with args.annotation.open() as f:
            data = json.load(f)
            if isinstance(data, list):
                # Find matching frame
                for frame in data:
                    if frame.get("image_path") == str(args.rgb):
                        bboxes = frame.get("objects", [])
                        break
            elif isinstance(data, dict):
                # COCO format
                pass  # TODO: Parse COCO format
        
        print(f"✅ Loaded {len(bboxes)} annotations")
    
    # Convert to point cloud
    print("🔄 Converting depth to point cloud...")
    intrinsics = None
    if args.camera_intrinsics:
        intrinsics = load_camera_intrinsics(args.camera_intrinsics)
    
    points, colors = depth_to_pointcloud(rgb_img, depth_img, intrinsics)
    print(f"✅ Point cloud: {len(points)} points")
    
    # Visualize
    print("🎨 Visualizing...")
    try:
        if OPEN3D_AVAILABLE:
            visualize_with_open3d(points, colors, bboxes)
        else:
            visualize_with_pyvista(points, colors, bboxes)
    except ImportError as e:
        print(f"❌ {e}")
        print("💡 Cài đặt một trong các thư viện:")
        print("   pip install open3d")
        print("   pip install pyvista")


if __name__ == "__main__":
    main()
