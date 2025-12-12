Loan Risk Assessment and Value Optimization

Project Overview

This repository contains the end-to-end analysis for automated loan decision-making using historical data (2007-2018).

The core objective is to move beyond simple default prediction (risk mitigation) toward profit optimization by comparing standard classification models against advanced sequential decision-making policies (Reinforcement Learning).

Key Tasks Completed (As per the Notebooks)

Data Exploration & Cleaning: Comprehensive EDA and handling of missing values.

Feature Engineering: Creation of new predictive features (e.g., debt-to-income ratios).

Model Building: Training and evaluation of a Deep Learning Predictor (Task 2) and an Offline RL Agent (Task 3).

Strategic Comparison (Task 4): Comparative analysis of statistical performance (AUC) vs. financial utility (Simulated Profit).

Files in this Repository

File Name

Description

loan_analysis.ipynb

The main Jupyter Notebook containing all data cleaning, EDA, model training, and final comparison logic.

requirements.txt

Lists all necessary Python libraries for recreating the environment.

README.md

This overview document.

.gitignore

Standard file to ignore temporary, cache, and large data files.

Setup and Installation

Clone the repository:

git clone [YOUR_REPOSITORY_URL]


Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate   # On Windows


Install dependencies:

pip install -r requirements.txt


Launch Jupyter Lab/Notebook and open loan_analysis.ipynb.
