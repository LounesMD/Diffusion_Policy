"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""
import argparse
from pathlib import Path
from typing import Tuple

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from utils.utils import (
    generate_brownian_step,
    generate_random_position,
    noise_image,
    patch_image,
)


def main() -> Tuple:
    parser = argparse.ArgumentParser(
        description="""A script to evaluate a policy on pusht with different cases.""",
    )

    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=6,
        help="Size of the number of predicted actions at once (n_action_steps). Note: n_action_steps <= horizon - n_obs_steps + 1.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=0,
        help="""Size of random (black) patch added in the image to deterior the input image quality.""",
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        default="outputs/eval/example_pusht_diffusion",
        help="""Directory to store the video of the evaluation.""",
    )

    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=300,
        help="""Maximum number of steps per episode.""",
    )

    parser.add_argument(
        "--policy_path",
        type=str,
        default="lerobot/diffusion_pusht",
        help="""Path to a policy to evaluate.""",
    )

    parser.add_argument(
        "--nb_simulation",
        type=int,
        default=1,
        help="""Total number of simulation to perform.""",
    )
    parser.add_argument(
        "--exp_prefix",
        type=str,
        default="prefix_",
        help="""Prefix of the experiment. Used when saving the results of the experiment.""",
    )
    parser.add_argument(
        "--exp_file",
        type=str,
        default="outputs/results.csv",
        help="""File where the results are saved. They will be saved.""",
    )

    parser.add_argument(
        "--patch_movement",
        type=str,
        default="random",
        help="""Defines how the black patch moves at each iteration. Could be random or Brownian.""",
    )

    parser.add_argument(
        "--image_quality",
        type=float,
        default=0.0,
        help="""Defines the std of the gaussian noise applied to the image.""",
    )

    parser.add_argument(
        "--n_obs_steps",
        type=int,
        default=2,
        help="""Number of observations to use. Note: n_action_steps <= horizon - n_obs_steps + 1.""",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="""Horizon: The diffusion model generates `horizon` steps worth of actions. Note: n_action_steps <= horizon - n_obs_steps + 1.""",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="""Defines the number of diffuion steps. Default number is 100 .""",
    )

    args = parser.parse_args()

    # Create a directory to store the video of the evaluation
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Download the diffusion policy for pusht environment
    pretrained_policy_path = Path(snapshot_download(args.policy_path))
    # OR uncomment the following to evaluate a policy from the local outputs/train folder.
    # pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    policy.config.n_action_steps = args.n_action_steps
    policy.config.n_obs_steps = args.n_obs_steps
    policy.config.horizon = args.horizon
    policy.diffusion.num_inference_steps = args.num_inference_steps
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(
            f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU."
        )
        # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
    policy.to(device)
    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=args.max_episode_steps,
    )

    results = []
    seeds = [
        80,
        546,
        765,
        931,
        665,
        147,
        708,
        669,
        23,
        927,
        549,
        35,
        303,
        820,
        276,
        212,
        24,
        451,
        225,
        758,
    ]
    for idx in range(args.nb_simulation):
        # policy.config.n_action_steps = args.n_action_steps
        res = run_simulation(
            env=env,
            policy=policy,
            patch_size=args.patch_size,
            device=device,
            output_directory=output_directory,
            iteration=idx,
            patch_movement=args.patch_movement,
            image_quality=args.image_quality,
            seed=seeds[idx],
        )
        results.append(res)

    # Prepare data for DataFrame
    columns = {
        f"{args.exp_prefix}col_{i+1}": sublist for i, sublist in enumerate(results)
    }

    # Convert to DataFrame, filling empty cells with NaN
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in columns.items()]))

    # Save to CSV
    df.to_csv(args.exp_file, index=False)


def run_simulation(
    env: gym.Env,
    policy: DiffusionPolicy,
    patch_size: int,
    device: torch.device,
    output_directory: Path,
    iteration: int,
    patch_movement: str,
    image_quality: float,
    seed=int,
):
    """
    Runs a single simulation episode in the given environment using the specified policy,
    applies a black patch to the image observations, collects rewards and frames, and saves
    the episode as a video file.

    Returns a list of tuples containing simulation metadata (iteration, step, reward, terminated).
    """

    # Reset the policy and environmens to prepare for rollout
    policy.reset()
    numpy_observation, _ = env.reset(seed=seed)
    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []
    # Patch the image
    image = torch.from_numpy(numpy_observation["pixels"])
    height, width, _ = image.shape
    patch_path = [
        (
            np.random.randint(10, height - patch_size - 10),
            np.random.randint(10, width - patch_size - 10),
        )
    ]

    image = patch_image(
        image=image, patch_size=patch_size, device=device, position=patch_path[0]
    )

    image = noise_image(image=image, noise_std_dev=image_quality)

    image_tensor_permuted = image.permute(1, 2, 0)
    image_numpy = image_tensor_permuted.cpu().numpy()
    frames.append(image_numpy)

    step = 0
    done = False
    res = []
    distance_traveled = [0]
    current_pos = numpy_observation["agent_pos"]
    while not done:
        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(numpy_observation["agent_pos"])
        image = torch.from_numpy(numpy_observation["pixels"])
        state = state.to(torch.float32)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)

        image = patch_image(
            image=image, patch_size=patch_size, device=device, position=patch_path[-1]
        )
        image = noise_image(image=image, noise_std_dev=image_quality)

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
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()
        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, _ = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        image_tensor_permuted = image.squeeze(0).permute(1, 2, 0)
        image_numpy = image_tensor_permuted.cpu().numpy()
        # image_numpy = (image_numpy).astype(np.uint8)
        frames.append(image_numpy)

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

        distance_traveled.append(  # Not used in this commit, but can be manually used of the reward.
            distance_traveled[-1]
            + numpy.linalg.norm(numpy_observation["agent_pos"] - current_pos)
        )
        current_pos = numpy_observation["agent_pos"]

        res.append((iteration, step, reward, terminated))

        if patch_movement == "random":
            patch_path.append(
                generate_random_position(
                    width=width, height=height, patch_size=patch_size
                )
            )
        elif patch_movement == "brownian":
            patch_path.append(
                generate_brownian_step(
                    current_position=patch_path[-1],
                    width=width,
                    height=height,
                    step_std_dev=5.0,
                )
            )

    if terminated:
        print("Success!")
    else:
        print("Failure!")
    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]
    # Encode all frames into a mp4 video.
    video_path = output_directory / ("_rollout_" + str(iteration) + ".mp4")
    frames = [
        (frame * 255).astype(np.uint8) for frame in frames
    ]  # Scale and convert all frames
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")
    return res


if __name__ == "__main__":
    main()
