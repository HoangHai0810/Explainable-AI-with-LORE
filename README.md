# Explainable AI with LORE

This project implements the **LORE** (LOcal Rule-based Explanations) method to provide intuitive explanations for decisions made by black-box machine learning models.

## Introduction

As machine learning models become increasingly complex, understanding their decision-making processes becomes essential. The **Explainable AI with LORE** project aims to:

- Provide local explanations for predictions of black-box models by extracting simple, interpretable rules.
- Generate neighborhood instances around a given sample to build a rule-based explanation model.
- Offer visualization tools to help users understand the reasons behind model decisions.

## Requirements

- **Python:** Version 3.6 or higher.
- **Dependencies:**  
  - numpy  
  - pandas  
  - scikit-learn  
  - matplotlib  
  - (Other libraries as needed, listed in the `requirements.txt` file)

## Installation

#### Clone the Repository
```bash
git clone https://github.com/HoangHai0810/Explainable-AI-with-LORE.git
cd Explainable-AI-with-LORE
```
#### Create and Activate a Virtual Environment (recommended)
```bash
python -m venv env
# Activate the virtual environment:
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```
#### Install the Required Libraries
```bash
pip install -r requirements.txt
```
## Usage
#### Prepare Your Data
Ensure your input data is preprocessed (normalized, encoded, etc.) and placed in the folder.
#### Run the Main Program
For example, to generate explanations for a specific data sample, run command cells in file `Explainable AI with LORE.ipynb`
#### View the Results
The generated explanations will be display as text outputs.
## Example
Suppose you have a dataset containing customer information and a model that predicts credit approval. Running the program may yield an explanation like:
``If income > 50 million and years of employment > 3, then the probability of credit approval is high.``
#### Contact
For any questions or suggestions, please contact:

Author: Hoàng Hải Anh Email: hoanghaianh0810@gmail.com
