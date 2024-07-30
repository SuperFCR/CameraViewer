import os

import numpy as np
import quaternion
import torch
from util.camera_pose_visualizer import CameraPoseVisualizer

if __name__ == '__main__':
    dir_path = 'c2ws'
    gt_poses_path = os.path.join(dir_path, 'gt_c2ws.pt')
    our_poses_path = os.path.join(dir_path, 'ours_c2ws.pt')
    compare_poses_path = os.path.join(dir_path, 'compare_c2ws.pt')

    gt_poses = load_poses(gt_poses_path)
    our_poses = load_poses(our_poses_path)
    compare_poses = load_poses(compare_poses_path)
    # gt_poses: torch.Size([25, 4, 4]), our_poses: torch.Size([25, 4, 4]), compare_poses: torch.Size([25, 4, 4])
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])

    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    visualizer.extrinsic2pyramid(np.eye(4), 'c', 10)

    visualizer.show()
