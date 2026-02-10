# Efficient Unsupervised Environment Design through Hierarchical Policy Representation Learning

[![arXiv](https://img.shields.io/badge/arXiv-2310.00301-b31b1b.svg)](https://arxiv.org/abs/2310.00301)

## 📋 Overview

This repository presents a novel hierarchical MDP framework for **Unsupervised Environment Design (UED)** under resource constraints. Unlike traditional UED approaches that rely on random environment generation, our method employs a teacher-student framework where an upper-level RL teacher agent intelligently generates training environments at the frontier of a lower-level student agent's capabilities.

### Key Features

- 🎯 **Resource-Efficient Training**: Designed for scenarios with limited computational resources and constraints on the number of generated environments
- 🏗️ **Hierarchical Framework**: Two-level MDP structure with teacher and student agents
- 🧠 **Intelligent Curriculum Generation**: Leverages previously discovered environment structures to create progressively challenging scenarios
- 📊 **Trajectory-Based Generation**: Uses generative trajectory modeling to understand and extend student capabilities
- 🚀 **Zero-Shot Transfer**: Trains agents with strong generalization capabilities for unseen environments

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hierarchical-ued-gtm.git
cd hierarchical-ued-gtm

# Create a virtual environment
conda create -n ued-gtm python=3.9
conda activate ued-gtm

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training the Teacher-Student Framework

```python
from ued_gtm import TeacherAgent, StudentAgent, HierarchicalTrainer

# Initialize teacher and student agents
teacher = TeacherAgent(obs_dim=..., action_dim=...)
student = StudentAgent(obs_dim=..., action_dim=...)

# Create hierarchical trainer
trainer = HierarchicalTrainer(
    teacher=teacher,
    student=student,
    max_environments=1000,  # Resource constraint
    curriculum_strategy='frontier'
)

# Train
trainer.train(num_iterations=10000)
```

### Generating Environments

```python
# Generate environments based on student's current capabilities
environments = teacher.generate_environments(
    student_representation=student.get_representation(),
    num_envs=10,
    difficulty_level='frontier'
)
```

### Evaluating Zero-Shot Transfer

```python
from ued_gtm import evaluate_transfer

# Evaluate on held-out test environments
results = evaluate_transfer(
    agent=student,
    test_envs=test_environment_set,
    num_episodes=100
)

print(f"Transfer Success Rate: {results['success_rate']:.2%}")
```

## 📊 Methodology

### Hierarchical MDP Framework

Our approach formulates environment design as a hierarchical Markov Decision Process:

- **Upper-Level (Teacher)**: Observes student policy representations and generates suitable training environments
- **Lower-Level (Student)**: Learns to solve the generated environments and develops general capabilities

### Generative Trajectory Modeling

The teacher agent uses trajectory modeling to:
1. Analyze student's performance patterns across different environments
2. Identify the frontier of student's capabilities
3. Generate new environments that challenge the student appropriately
4. Reuse and recombine structures from previously successful environments

## 📈 Results

Our method demonstrates significant improvements over baseline UED approaches:

- **Sample Efficiency**: Achieves comparable performance with 40% fewer generated environments
- **Transfer Performance**: Improves zero-shot transfer success rate by 15-20% on benchmark tasks
- **Resource Utilization**: More effective use of computational budget through intelligent environment selection

## 🏗️ Repository Structure

```
hierarchical-ued-gtm/
├── ued_gtm/
│   ├── agents/
│   │   ├── teacher.py          # Teacher agent implementation
│   │   └── student.py          # Student agent implementation
│   ├── envs/
│   │   ├── env_generator.py    # Environment generation utilities
│   │   └── base_env.py         # Base environment class
│   ├── models/
│   │   ├── trajectory_model.py # Generative trajectory modeling
│   │   └── networks.py         # Neural network architectures
│   └── training/
│       ├── hierarchical_trainer.py
│       └── curriculum.py       # Curriculum generation logic
├── experiments/
│   ├── configs/               # Experiment configurations
│   └── run_experiments.py     # Main experiment script
├── tests/
│   └── test_*.py             # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## 🏃‍♂️ Run Commands

### 1. **Random Training**:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --seed 91 --num_budget 20 --num_udpates_per_env 50 --buffer_size 10000 --cv False --newMDP False --gamma 1 --domain_randomization True --accel False --cv False --num_env 2 --diffusion_synth_buffer_size 900 --diffusion_lr 1e-3 --diffusion_max_state 200 --use_diffusion False --num_eval_envs 10
```

### 2. **Accel Training**:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --seed 91 --num_budget 20 --num_udpates_per_env 50 --buffer_size 10000 --cv False --newMDP False --gamma 1 --domain_randomization False --accel True --cv False --num_env 2 --diffusion_synth_buffer_size 900 --diffusion_lr 1e-3 --diffusion_max_state 200 --use_diffusion False --num_eval_envs 10
```

### 3. **New MDP Training**:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --seed 91 --num_budget 20 --num_udpates_per_env 50 --buffer_size 10000 --cv False --newMDP True --gamma 1 --domain_randomization False --accel False --cv False --num_env 5 --diffusion_synth_buffer_size 900 --diffusion_lr 1e-3 --diffusion_max_state 200 --use_diffusion False --num_eval_envs 10
```

### 4. **Shed Training**:

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --seed 95 --num_budget 50 --num_udpates_per_env 20 --buffer_size 2048 --cv False --newMDP True --gamma 1 --domain_randomization False --accel False --cv False --num_env 5 --diffusion_synth_buffer_size 900 --diffusion_lr 1e-3 --diffusion_max_state 200 --use_diffusion True --num_eval_envs 10
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{li2023enhancing,
  title={Enhancing the Hierarchical Environment Design via Generative Trajectory Modeling},
  author={Li, Dexun and Varakantham, Pradeep},
  journal={arXiv preprint arXiv:2310.00301},
  year={2023}
}
```

---
