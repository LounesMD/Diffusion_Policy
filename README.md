# Diffusion Policy: MVA Robotics Course - Project

This repository contains the code for the project of the Robotics course of the MVA master at ENS Paris-Saclay.
<p align="center">
  <img src="./report/figures/trajectories_multimodality.png" width="275" height="250" title="Multimodel generation">
</p>

## Installation

First, you need to clone [`lerobot`](https://github.com/huggingface/lerobot) library source code from the github repository:

```bash
git clone https://github.com/huggingface/lerobot.git
```

Then, install the dependencies:

```bash
conda env create -f environment.yaml
conda activate robotics_project
```

## Usage

You can run any of the scripts in the `src` folder from the root of the repository.

### Visualizing the PushT Dataset Samples

```bash
python src/test_dataset.py
```

### Evaluating a Policy in the Sandbox
To evaluate and run a pushT policy, run the following command:
```bash
python src/evaluation_sandbox.py --output_directory outputs/eval/example_pusht_diffusion --max_episode_steps 300 --num_inference_steps 100 --policy_path lerobot/diffusion_pusht --nb_simulation 1 --exp_prefix prefix_ --exp_file outputs/results.csv --n_action_steps 6 --patch_size 0 --horizon 16 --n_obs_steps 2
```
Then, you can visualize the results in `./outputs/` with `results.csv`, and `eval/_example_pusht_diffusion/*.mp4`

## Acknowledgements

We would like to thank:
* The authors of the paper [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/) from which we are basing this project.
* The authors of the [`lerobot`](https://github.com/huggingface/lerobot) library for providing a clean and easy-to-use implementation of the original paper, as well as environments and tools for training and testing the models.

## Contact and Contributors

This project is conducted by: [Tom Ravaud]() and [Lounès Meddahi]().