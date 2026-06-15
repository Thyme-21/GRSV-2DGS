import os
import sys
import json
import argparse
from pathlib import Path

# Linux headless rendering. Must be set before importing pyrender.
if sys.platform.startswith("linux") and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio


def load_camera_json(camera_json_path):
    """
    支持常见两类格式：

    1. 3DGS / 2DGS cameras.json:
    [
        {
            "id": 0,
            "img_name": "...",
            "width": 1008,
            "height": 756,
            "position": [...],
            "rotation": [[...], [...], [...]],
            "fy": ...,
            "fx": ...
        }
    ]

    2. Blender/NeRF transforms 格式:
    {
        "camera_angle_x": ...,
        "frames": [
            {
                "file_path": "...",
                "transform_matrix": [[...], ...]
            }
        ]
    }
    """
    with open(camera_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def find_camera(data, camera_id):
    """
    camera_id 可以是：
    - 数字 id，例如 0, 4, 15
    - img_name，例如 DSC_0004
    - frame 下标，例如 0, 1, 2
    """
    camera_id_str = str(camera_id)

    # 3DGS / 2DGS cameras.json: list[dict]
    if isinstance(data, list):
        # 优先按 id 匹配
        for cam in data:
            if "id" in cam and str(cam["id"]) == camera_id_str:
                return cam, "3dgs"

        # 再按 img_name 匹配
        for cam in data:
            if "img_name" in cam and str(cam["img_name"]) == camera_id_str:
                return cam, "3dgs"

        # 最后按 list 下标匹配
        idx = int(camera_id)
        if 0 <= idx < len(data):
            return data[idx], "3dgs"

        raise ValueError(f"Cannot find camera_id={camera_id} in camera json.")

    # Blender/NeRF transforms 格式
    if isinstance(data, dict) and "frames" in data:
        frames = data["frames"]

        # 按 file_path 匹配
        for frame in frames:
            if "file_path" in frame and str(frame["file_path"]) == camera_id_str:
                return {"frame": frame, "root": data}, "blender"

        # 按下标匹配
        idx = int(camera_id)
        if 0 <= idx < len(frames):
            return {"frame": frames[idx], "root": data}, "blender"

        raise ValueError(f"Cannot find frame={camera_id} in transforms json.")

    raise ValueError("Unsupported camera.json format.")


def build_camera_from_3dgs(cam, flip_yz=True):
    """
    读取 3DGS / 2DGS cameras.json 中的相机。

    3DGS cameras.json 中通常保存的是 camera-to-world:
        rotation: 3x3
        position: 3

    pyrender 使用 OpenGL 相机坐标：
        x right, y up, z backward

    COLMAP / 3DGS 通常近似为 OpenCV 相机坐标：
        x right, y down, z forward

    所以默认需要对相机坐标的 y/z 轴翻转。
    """
    width = int(cam["width"])
    height = int(cam["height"])

    fx = float(cam["fx"])
    fy = float(cam["fy"])

    cx = float(cam.get("cx", width / 2.0))
    cy = float(cam.get("cy", height / 2.0))

    rotation = np.array(cam["rotation"], dtype=np.float64)
    position = np.array(cam["position"], dtype=np.float64)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position

    if flip_yz:
        # OpenCV camera coordinates -> OpenGL camera coordinates
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        c2w = c2w @ cv_to_gl

    camera = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=0.01,
        zfar=1000.0,
    )

    return camera, c2w, width, height


def build_camera_from_blender(cam_pack, flip_yz=False):
    """
    读取 Blender/NeRF transforms.json。
    Blender transforms 通常已经是 OpenGL 风格 camera-to-world，
    所以默认不 flip_yz。
    """
    root = cam_pack["root"]
    frame = cam_pack["frame"]

    c2w = np.array(frame["transform_matrix"], dtype=np.float64)

    width = int(root.get("w", root.get("width", 800)))
    height = int(root.get("h", root.get("height", 800)))

    if "fl_x" in root and "fl_y" in root:
        fx = float(root["fl_x"])
        fy = float(root["fl_y"])
    else:
        camera_angle_x = float(root["camera_angle_x"])
        fx = 0.5 * width / np.tan(0.5 * camera_angle_x)
        fy = fx

    cx = float(root.get("cx", width / 2.0))
    cy = float(root.get("cy", height / 2.0))

    if flip_yz:
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        c2w = c2w @ cv_to_gl

    camera = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=0.01,
        zfar=1000.0,
    )

    return camera, c2w, width, height


def load_mesh(mesh_path, color=(0.75, 0.75, 0.75, 1.0)):
    """
    读取 mesh。

    关键点：
    不做 normalize
    不做 center
    不做 scale
    不做 rotate
    """
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    if mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty: {mesh_path}")

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.8,
        baseColorFactor=color,
    )

    render_mesh = pyrender.Mesh.from_trimesh(
        mesh,
        material=material,
        smooth=False,
    )

    return render_mesh


def render_mesh(
    mesh_path,
    camera_json_path,
    camera_id,
    output_path,
    bg_color=(1.0, 1.0, 1.0, 1.0),
    mesh_color=(0.75, 0.75, 0.75, 1.0),
    flip_yz=True,
    save_depth=False,
):
    data = load_camera_json(camera_json_path)
    cam_data, cam_type = find_camera(data, camera_id)

    if cam_type == "3dgs":
        camera, camera_pose, width, height = build_camera_from_3dgs(
            cam_data,
            flip_yz=flip_yz,
        )
    elif cam_type == "blender":
        camera, camera_pose, width, height = build_camera_from_blender(
            cam_data,
            flip_yz=False,
        )
    else:
        raise ValueError(f"Unsupported camera type: {cam_type}")

    scene = pyrender.Scene(
        bg_color=bg_color,
        ambient_light=(0.4, 0.4, 0.4),
    )

    mesh_node = load_mesh(mesh_path, color=mesh_color)
    scene.add(mesh_node, pose=np.eye(4))

    scene.add(camera, pose=camera_pose)

    # 加一个跟随相机的光源，避免 mesh 太暗
    light = pyrender.DirectionalLight(
        color=np.ones(3),
        intensity=3.0,
    )
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height,
    )

    color, depth = renderer.render(scene)
    renderer.delete()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(output_path, color)

    if save_depth:
        depth_path = output_path.with_name(output_path.stem + "_depth.npy")
        np.save(depth_path, depth)
        print(f"Saved depth to: {depth_path}")

    print(f"Saved mesh rendering to: {output_path}")
    print(f"Camera type: {cam_type}")
    print(f"Resolution: {width} x {height}")
    print(f"Camera id: {camera_id}")
    print("Note: mesh was rendered without individual centering, scaling, or normalization.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to mesh file, e.g. mesh.ply or mesh.obj",
    )

    parser.add_argument(
        "--camera_json",
        type=str,
        required=True,
        help="Path to cameras.json / camera.json / transforms.json",
    )

    parser.add_argument(
        "--camera_id",
        type=str,
        required=True,
        help="Camera id, img_name, or frame index",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output image path, e.g. render.png",
    )

    parser.add_argument(
        "--no_flip_yz",
        action="store_true",
        help="Disable OpenCV-to-OpenGL camera conversion. Use this if viewpoint looks wrong.",
    )

    parser.add_argument(
        "--save_depth",
        action="store_true",
        help="Also save rendered depth as .npy",
    )

    args = parser.parse_args()

    render_mesh(
        mesh_path=args.mesh,
        camera_json_path=args.camera_json,
        camera_id=args.camera_id,
        output_path=args.output,
        flip_yz=not args.no_flip_yz,
        save_depth=args.save_depth,
    )


if __name__ == "__main__":
    main()


# python render_mesh_from_camera_json.py --mesh kitchen/DGS.ply --camera_json camera_34.json --camera_id 34 --output output/mesh_view_34.png