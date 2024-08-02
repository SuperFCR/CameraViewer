import os

import numpy as np
import torch
from util.camera_pose_visualizer import CameraPoseVisualizer


# Function to load poses from a file
def load_poses(path):
    return torch.load(path).numpy()

# Function to extract rotation matrix and translation vector from a pose matrix
def decompose_pose(pose):
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return rotation, translation

# Function to swap Y and Z axes
def swap_yz(rotation, translation):
    # Swap rows 1 and 2 in the rotation matrix
    rotation = rotation[:, [0, 2, 1]]
    rotation[:, 1] = -rotation[:, 1]
    
    # Swap Y and Z in the translation vector
    translation = translation[[0, 2, 1]]
    translation[1] = -translation[1]  # Invert new Y to maintain orientation
    return rotation, translation

def rgb_to_normalized(rgb):
    return [c / 255 for c in rgb]

# Main script
if __name__ == '__main__':

    compare_face_color = rgb_to_normalized([204, 234, 239])
    compare_edge_color = rgb_to_normalized([150, 190, 210])
    # compare_edge_color = rgb_to_normalized([4,157,217])
    # compare_edge_color = rgb_to_normalized([242, 135, 5])

    gt_face_color = rgb_to_normalized([235, 214, 232])
    # gt_edge_color = rgb_to_normalized([183, 101, 167])
    # gt_edge_color = rgb_to_normalized([242, 135, 5])
    # gt_edge_color = rgb_to_normalized([110,30,130])
    gt_edge_color = rgb_to_normalized([151,141,242])
    # gt_edge_color = rgb_to_normalized([242, 135, 5])
    # gt_edge_color = rgb_to_normalized([255, 187,111])

    z_scale = 1.0
    trans_scale = 5

    dir_path = 'greate-nerf'
    gt_poses_path = os.path.join(dir_path, 'gt_camera_poses.pt')
    our_poses_path = os.path.join(dir_path, 'ours_camera_poses.pt')
    colmap_poses_path = os.path.join(dir_path, 'colmap_camera_poses.pt')

    gt_poses = load_poses(gt_poses_path)
    our_poses = load_poses(our_poses_path)
    compare_poses = load_poses(colmap_poses_path)
    gt_poses[:,1,3] *= z_scale
    our_poses[:,1,3] *= z_scale
    compare_poses[:,1,3] *= z_scale
    # gt_poses: torch.Size([25, 4, 4]), our_poses: torch.Size([25, 4, 4]), compare_poses: torch.Size([25, 4, 4])
    
    # Initialize the visualizer with the axis limits for x, y, and z
    visualizer = CameraPoseVisualizer([-50,50], [-50, 50], [-20 , 20])



    # Flag to choose between 'our' and 'compare' poses
    flag = 'compare'  # 'our' or 'compare'

    gt_translation_vectors = []
    translation_vectors = []

    # Add GT poses to the visualizer
    for pose in gt_poses:
        rotation, translation = decompose_pose(pose)
        # rotation, translation = swap_yz(rotation, translation)  # Swap Y and Z axes
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation
        extrinsic_matrix[:3, 3] = translation * trans_scale
        visualizer.extrinsic2pyramid(extrinsic_matrix, gt_edge_color, 10)  # 'c' stands for cyan color
        gt_translation_vectors.append(translation * trans_scale)
    
    # Add Our poses to the visualizer
    if flag == 'our':
        for pose in our_poses:
            rotation, translation = decompose_pose(pose)
            # rotation, translation = swap_yz(rotation, translation)  # Swap Y and Z axes
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = rotation
            extrinsic_matrix[:3, 3] = translation * trans_scale
            visualizer.extrinsic2pyramid(extrinsic_matrix, compare_edge_color, 10)  # 'r' stands for red color
            translation_vectors.append(translation * trans_scale)

    # Add Compare poses to the visualizer
    elif flag == 'compare':
        for pose in compare_poses:
            rotation, translation = decompose_pose(pose)
            # rotation, translation = swap_yz(rotation, translation)  # Swap Y and Z axes
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = rotation
            extrinsic_matrix[:3, 3] = translation * trans_scale
            visualizer.extrinsic2pyramid(extrinsic_matrix, compare_edge_color , 10)  # 'g' stands for green color
            translation_vectors.append(translation * trans_scale)

    else:
        assert "Invalid flag"
    
    # Plot error lines between GT and Compare poses
    for gt_translation, translation in zip(gt_translation_vectors, translation_vectors):
        visualizer.plot_error_line(gt_translation,translation, 'r')

    # visualizer.show(name = 'gt pose vs. compared pose')
    visualizer.show(name =f'gt pose vs. {flag} pose')
