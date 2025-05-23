# DDRS: Disease Diagnosis and Recommendation System

This repository contains a machine learning-based system for predicting diseases and providing recommendations. The project includes models for diabetes and heart disease prediction, along with a general disease prediction model.

## Project Structure
DDRS/ ├── models training/ │ ├── api.py │ ├── Diabetes.ipynb │ ├── Disease_prediction.ipynb │ ├── Heart_Disease.ipynb │ ├── datasets/ │ │ ├── diabetes.csv │ │ ├── general.csv │ │ ├── heart.csv │ ├── savedmodels/ │ │ ├── diabetes_model.sav │ │ ├── heartdisease.pkl │ │ ├── generalmodel/ │ │ ├── best_model.pkl │ │ ├── scaler.pkl ├── python&documentation/ │ ├── project documentation.pdf ├── README.md


### Key Components

#### 1. **Datasets**
- `diabetes.csv`: Dataset for diabetes prediction.
- `general.csv`: General dataset for disease prediction.
- `heart.csv`: Dataset for heart disease prediction.

#### 2. **Notebooks**
- `Diabetes.ipynb`: Notebook for training and evaluating diabetes prediction models.
- `Heart_Disease.ipynb`: Notebook for training and evaluating heart disease prediction models.
- `Disease_prediction.ipynb`: General disease prediction notebook.

#### 3. **Saved Models**
- `diabetes_model.sav`: Trained model for diabetes prediction.
- `heartdisease.pkl`: Trained model for heart disease prediction.
- `generalmodel/`: Contains the best general disease prediction model (`best_model.pkl`) and its scaler (`scaler.pkl`).

#### 4. **API**
- `api.py`: API implementation for serving the trained models.

#### 5. **Documentation**
- `project documentation.pdf`: Detailed documentation of the project.

## How to Use

1. **Train Models**: Use the Jupyter notebooks (`Diabetes.ipynb`, `Heart_Disease.ipynb`, `Disease_prediction.ipynb`) to train and evaluate the models.
2. **Run API**: Use `api.py` to serve the trained models for predictions.
3. **Datasets**: Ensure the datasets are placed in the `datasets/` directory for training.

## Requirements

- Python 3.11.9
- Required libraries: `scikit-learn`, `pandas`, `numpy`, `flask`, etc.

Install dependencies using:
```bash
pip install -r requirements.txt
```
### License
This project is licensed under the MIT License.

