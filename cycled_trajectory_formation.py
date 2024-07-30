import torch


def generate_camera_poses(radius, height, num_cameras):
    poses = []
    for i in range(num_cameras):
        angle = 2 * torch.pi * i / num_cameras
        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)
        z = torch.tensor(height, dtype=torch.float32)
        
        # Translation vector (camera position)
        translation = torch.tensor([x, y, z], dtype=torch.float32)
        
        # Camera looks towards the origin, so the forward vector is -translation
        forward = -translation / torch.norm(translation)
        # Assume up vector is along z-axis
        up = torch.tensor([0, 0, 1], dtype=torch.float32)
        
        # Right vector (cross product of up and forward vectors)
        right = torch.cross(up, forward)
        right /= torch.norm(right)
        
        # Recalculate up vector
        up = torch.cross(forward, right)
        
        # Rotation matrix
        rotation = torch.eye(4, dtype=torch.float32)
        rotation[:3, 0] = right
        rotation[:3, 1] = up
        rotation[:3, 2] = forward
        
        # Extrinsic matrix
        extrinsic = torch.eye(4, dtype=torch.float32)
        extrinsic[:3, :3] = rotation[:3, :3]
        extrinsic[:3, 3] = translation
        
        poses.append(extrinsic)
        
    return torch.stack(poses)

# Parameters
N = 40  # Number of cameras
radius = 10.0  # Radius of the circle
height = 5.0  # Height at which cameras are placed

# Generate camera poses using PyTorch
camera_poses = generate_camera_poses(torch.tensor(radius, dtype=torch.float32), torch.tensor(height, dtype=torch.float32), N)

# Output the poses
print(camera_poses)