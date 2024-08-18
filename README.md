# Variance Reduced Policy Gradient Method for Multi-Objective Reinforcement Learning

Multi-Objective Policy Gradient Methods in Gymnasium Environments

## Overview

This is the official repository to the paper "Variance Reduced Policy Gradient Method
for Multi-Objective Reinforcement Learning". We implement two multi-objective reinforcement learning algorithms, namely `MOPG` (Multi-Objective Policy Gradient) and `MOTSIVRPG` (Multi-Objective Truncated Stochastic Incremental Variance-Reduced Policy Gradient). The project is designed to work within various single-objective and multi-objective environments in the Gymnasium framework. Additionally we implement the Server Queues environment, traditionally used to benchmark MORL algorithms. The code facilitates training of these algorithms in parallel or sequential mode, with support for environment-specific configurations.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Configuration](#configuration)
   - [Training](#training)
3. [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```
Make sure to use Python 3.12.3 or higher.

## Usage

### Configuration

All configurations for the algorithms and environments are specified in the `Config` class within the code. Key configuration options include:

- `alg_name`: Algorithm to use (`'MOPG'` or `'MOTSIVRPG'`).
- `environment`: Environment to use (`'queue'`, `'deep-sea-treasure-v0'`, `'Acrobot-v1'`, `'CartPole-v1'`).
- `num_runs`: Number of experiments to run.
- `parallel`: Whether to run experiments in parallel.
- `epochs`: Number of training epochs.
- `checkpoint`: Whether to save intermediate training results.
- `checkpoint_interval`: Interval at which to save checkpoints.
- `debug`: Whether to print debugging information.

The policy network parameters can be found within the same file.

### Training

To start training, simply execute the script:

```bash
python main.py
```

This will initialize the environment and train the selected algorithm for the specified number of epochs.

The results are stored under the `results` directory with filenames that include the environment and algorithm names, along with additional information such as the epoch and thread ID.

## License

This work is licensed under the Creative Commons Attribution 4.0 International License. See the [LICENSE](LICENSE) file for more details.
