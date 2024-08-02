import torch


def generate_camera_poses(radius, height, num_cameras):
    poses = []
    for i in range(num_cameras):
        angle = torch.tensor(2 * i * 3.14159 / num_cameras, dtype=torch.float32)
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


def generate_noised_camera_poses(radius, height, num_cameras, translation_noise_std=0.0, rotation_noise_std=0.0):
    poses = []
    for i in range(num_cameras):
        angle = torch.tensor(2 * i * 3.14159 / num_cameras, dtype=torch.float32)
        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)
        z = torch.tensor(height, dtype=torch.float32)
        
        # Translation vector (camera position)
        translation = torch.tensor([x, y, z], dtype=torch.float32)
        
        # Add Gaussian noise to translation
        if translation_noise_std > 0:
            noise = torch.normal(mean=0, std=translation_noise_std, size=translation.size())
            translation += noise
        
        # Camera looks towards the origin, so the forward vector is -translation
        forward = -translation / torch.norm(translation)
        # Assume up vector is along z-axis
        up = torch.tensor([0, 0, 1], dtype=torch.float32)
        
        # Right vector (cross product of up and forward vectors)
        right = torch.cross(up, forward)
        right /= torch.norm(right)
        
        # Recalculate up vector
        up = torch.cross(forward, right)
        
        # Construct rotation matrix
        rotation = torch.eye(4, dtype=torch.float32)
        rotation[:3, 0] = right
        rotation[:3, 1] = up
        rotation[:3, 2] = forward

        # Add small random rotations to the rotation matrix
        if rotation_noise_std > 0:
            # Generate small random angles for each axis
            angle_noise = torch.normal(mean=0, std=rotation_noise_std, size=(3,))
            Rx = torch.tensor([[1, 0, 0],
                               [0, torch.cos(angle_noise[0]), -torch.sin(angle_noise[0])],
                               [0, torch.sin(angle_noise[0]), torch.cos(angle_noise[0])]], dtype=torch.float32)
            Ry = torch.tensor([[torch.cos(angle_noise[1]), 0, torch.sin(angle_noise[1])],
                               [0, 1, 0],
                               [-torch.sin(angle_noise[1]), 0, torch.cos(angle_noise[1])]], dtype=torch.float32)
            Rz = torch.tensor([[torch.cos(angle_noise[2]), -torch.sin(angle_noise[2]), 0],
                               [torch.sin(angle_noise[2]), torch.cos(angle_noise[2]), 0],
                               [0, 0, 1]], dtype=torch.float32)
            
            # Apply the random rotations
            random_rotation = Rz @ Ry @ Rx
            rotation[:3, :3] = random_rotation @ rotation[:3, :3]

        # Extrinsic matrix
        extrinsic = torch.eye(4, dtype=torch.float32)
        extrinsic[:3, :3] = rotation[:3, :3]
        extrinsic[:3, 3] = translation
        
        poses.append(extrinsic)
        
    return torch.stack(poses)


# Parameters
N = 40  # Number of cameras
radius = 10.0  # Radius of the circle
height = 0.0  # Height at which cameras are placed
translation_noise_std = 0.5  # Standard deviation of Gaussian noise for translation
rotation_noise_std = 0.1  # Standard deviation of Gaussian noise for rotation angles (in radians)

# Generate camera poses using PyTorch
camera_poses = generate_camera_poses(torch.tensor(radius, dtype=torch.float32), torch.tensor(height, dtype=torch.float32), N)
noised_camera_poses = generate_noised_camera_poses(radius, height, N, translation_noise_std, rotation_noise_std)
our_cameara_poses = generate_noised_camera_poses(radius, height, N, translation_noise_std/5, rotation_noise_std/5)
# Output the poses
print(camera_poses.shape, noised_camera_poses.shape)
torch.save(camera_poses, 'greate-nerf/gt_camera_poses.pt')
torch.save(noised_camera_poses, 'greate-nerf/colmap_camera_poses.pt')
torch.save(our_cameara_poses, 'greate-nerf/ours_camera_poses.pt')