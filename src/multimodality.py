"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch
from huggingface_hub import snapshot_download
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.utils import populate_queues
from utils.utils import (
    patch_image,
    generate_random_position,
)


class DiffusionPolicyCustom(DiffusionPolicy):
    @torch.no_grad
    def sample_trajectories(
        self,
        batch: dict[str, Tensor],
        image: np.ndarray,
        n_trajectories: int = 2,
    ) -> Tensor:
        """Sample `n_trajectories` trajectories from the policy.

        Args:
            batch (dict[str, Tensor]): Batch of observations.
            n_trajectories (int): Number of trajectories to sample.
                Defaults to 2.

        Returns:
            Tensor: Sampled trajectories.
        """
        image = batch["observation.image"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Duplicate the batch `n_trajectories` times
        batch = {
            k: v.repeat_interleave(n_trajectories, dim=0) for k, v in batch.items()
        }
        
        # Normalize the inputs
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            # Shallow copy so that adding a key doesn't modify the original
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.expected_image_keys],
                dim=-4,
            )
        
        # Note: It's important that this happens after stacking the images into a
        # single key.
        self._queues = populate_queues(self._queues, batch)
        
        if len(self._queues["action"]) == 0:
            # Stack n latest observations from the queue
            batch = {
                k: torch.stack(
                    list(self._queues[k]),
                    dim=1,
                ) for k in batch if k in self._queues
            }
            # Predict the sequence of actions to take (not the
            # whole horizon)
            actions = self.diffusion.generate_actions(batch)
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # Add the actions to the queue
            self._queues["action"].extend(actions.transpose(0, 1))
        
        
        plt.figure()
        plt.imshow(image)
        
        for i in range(n_trajectories):
            traj = []
            for action in self._queues["action"]:
                traj.append(action[i].cpu().numpy())
        
            traj = np.array(traj)
            traj = np.int32(traj * 96 / 512)
            # traj = np.int32(traj * 680 / 512)
        
            plt.plot(traj[:, 0], traj[:, 1], "o-")
            
        # plt.legend()
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("batch_trajectories.png", bbox_inches="tight", pad_inches=0)
        plt.show()
        
        action = self._queues["action"].popleft()
        action = action[0]
        
        return action


# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Download the diffusion policy for pusht environment
# pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
pretrained_policy_path = Path("outputs/train/example_pusht_diffusion_multi")

policy = DiffusionPolicyCustom.from_pretrained(pretrained_policy_path)
policy.eval()

policy.config.n_action_steps = 8
# policy.config.horizon = 64
policy.diffusion.num_inference_steps = 100

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Device set to:", device)
else:
    device = torch.device("cpu")
    print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
    # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
    policy.diffusion.num_inference_steps = 10

policy.to(device)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

# Reset the policy and environmens to prepare for rollout
policy.reset()

# Set the environment to a specific state
numpy_observation, info = env.reset(options={"reset_to_state": [
    256 + 100 * np.sin(np.pi / 4),
    256 - 100 * np.cos(np.pi / 4),
    256,
    256,
    np.pi / 4,
]})
# numpy_observation, info = env.reset(seed=42)
env.block.position = [
    256 - 50 * np.sin(np.pi / 4),
    256 + 50 * np.cos(np.pi / 4),
]
env.block.angle = np.pi / 4

# print(env.space.damping)
# print(env.block.position)
# print(env.block.angle)


# Extract the image and state from the observation
image = torch.from_numpy(numpy_observation["pixels"])
height, width, _ = image.shape
patch_size = 0
patch_path = [
    (
        np.random.randint(patch_size, height - patch_size - patch_size),
        np.random.randint(patch_size, width - patch_size - patch_size),
    )
]

plt.figure()
plt.imshow(image)
plt.show()

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

step = 0
done = False
while not done:
    # Prepare observation for the policy running in Pytorch
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])
    
    print("State:", state)

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = patch_image(
        image=image, patch_size=patch_size, device=device, position=patch_path[-1]
    )
    patch_path.append(
        generate_random_position(
            width=width, height=height, patch_size=patch_size
        )
    )

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.sample_trajectories(
            observation,
            frames[-1],
            n_trajectories=16,
        )
        # action = policy.select_action(observation)
    
    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()
    
    # print(numpy_action.shape)
    numpy_action = np.array([
        256 + 50 * np.sin(np.pi / 4) + 0 * np.cos(np.pi / 4),
        256 - 50 * np.cos(np.pi / 4) + 0 * np.sin(np.pi / 4),
    ])

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")
    
    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reach (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    # done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
