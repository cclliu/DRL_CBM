# DRL CMB Project

This repository contains the code and resources for the DRL CMB Project, which focuses on developing and utilizing surrogate models and drl models for optimization tasks. Below is an overview of the directory structure and key files.

## Directory Structure

- **data**: Contains raw and processed data used in the project.
- **data_cleaning**: Scripts and tools for cleaning and preprocessing data.
- **draw**: Utilities for generating visualizations and plots.
- **drl**: Code related to Deep Reinforcement Learning (DRL) implementations.
- **figure**: directory for figures for DRL_CMB.
- **figures**: directory for figures for surrogate_CMB.
- **logs**: Log files for tracking experiments and model training.
- **models**: Saved model files and related configurations.
- **scaler**: Tools for data scaling and normalization.
- **surrogate_model**: Implementation of surrogate models.
- **utils**: Utility functions and helper scripts.

## Key Files

- **.gitignore**: Specifies files and directories to be ignored by Git.
- **config.py**: Python configuration file for project settings.
- **config.yaml**: YAML configuration file for project settings.
- **main.py**: Main script to surrogate_CMB.
- **\drl\traindqn.py**:Main script to DRL_CMB.
- **notebook.txt**: Notes and documentation from Jupyter notebooks.
- **problems.md**: Documentation of known issues and problems.
- **README.md**: This file, providing an overview of the project.
- **requirements.txt**: List of Python dependencies required for the project.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DRL_CMB.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Conduct Data Preprocessing
   ```bash
   python \data_cleaning\data_cleaning.py
   ```
4. Run the main script:
   ```bash
   python main.py
   ```
5. Run the \drl\traindqn.py script:
   ```bash
   python \drl\traindqn.py
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
# DRL_CBM
# DRL_CBM
# DQN_CBM
