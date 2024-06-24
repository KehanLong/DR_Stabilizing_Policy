Distributionally Robust Lyapunov-stable Policy Learning
========================================================================
This repository contains code for the work "Distributionally Robust Policy and Lyapunov-Certificate Learning".

# Clone Repo
```bash
git clone https://github.com/KehanLong/DR_Stabilizing_Policy
```
# Dependencies

Install the required dependcies by using a anaconda virtual environment:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate DR_Stabilizing_Policy
```

To formally verify the network: please install auto_LiRPA and alpha-beta-CROWN:

git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
(cd alpha-beta-CROWN/auto_LiRPA && pip install -e .)
(cd alpha-beta-CROWN/complete_verifier && pip install -r requirements.txt)

Finally, go to the project directory
```
cd DR_LF_Learning
```

## Acknowledgments

This project uses a modified version of the Gymnasium library, which is an open-source library for developing and comparing reinforcement learning algorithms. We would like to acknowledge the original work by the Farama Foundation and the contributors of the Gymnasium repository. The original repository can be found at https://github.com/Farama-Foundation/Gymnasium.

The Gymnasium_modified directory in this project contains a few minor modifications to the system dynamics of the Mountain Car and Inverted Pendulum environments. 


# Training

To re-train the provided examples, simply run
```python Inverted_pendulum_learning.py``` or ```python mountain_car_learning.py```

Pre-trained models are available in the saved_models/joint_clf_controller_models directory.

To reproduce the results, refer to the files ```Inverted_pendulum_evaluate```; ```Mountain_car_evaluate```. 

# Verification 


cd complete_verifier
python alpha-beta-CROWN/complete_verifier/abcrown.py --config DR_LF_Learning/test_config.yaml

# Visualization Results

The results of the learned DR Lyapunov-stable policy are demonstrated through the following GIFs:

## Inverted Pendulum

### Initial state 1

| Baseline | Distributionally Robust |
|:--------:|:-----------------------:|
| ![Baseline Initial State 1](Results/inverted_pendulum_baseline_case1.gif) | ![Distributionally Robust Initial State 1](Results/inverted_pendulum_DR_case1.gif) |

### Initial state 2

| Baseline | Distributionally Robust |
|:--------:|:-----------------------:|
| ![Baseline Initial State 2](Results/inverted_pendulum_baseline_case2.gif) | ![Distributionally Robust Initial State 2](Results/inverted_pendulum_DR_case2.gif) |

## Mountain Car

| Baseline | Distributionally Robust |
|:--------:|:-----------------------:|
| ![Baseline Mountain Car](Results/mountain_car_baseline.gif) | ![Distributionally Robust Mountain Car](Results/mountain_car_DR.gif) |

