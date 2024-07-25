
# Machine Learning Engineering Coding Best Practices
# Model Evaluation and Validation
## Project: Median housing value prediction

### Install

This project requires **Python** and the Python libraries can be installed:

- `pip install {{package_name}}`

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

### Run

In a terminal or command window, navigate to the top-level project directory `mle-training-/` (that contains this README) and run one of the following commands:

```powershell
ipython notebook MLE-Training Housing.ipynb
```  
or
```powershell
jupyter notebook
```
This will open the code and project file.

### Data

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

- Linear regression
- Decision Tree
- Random Forest

**Steps performed**
- We prepare and clean the data. We check and impute for missing values.
- Features are generated and the variables are checked for correlation.
- Multiple sampling techinuqies are evaluated. The data set is split into train and test.
- All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean   squared error.


# mle-training
MLE Training for Assignments

### Steps to run the Python code
```sh
git clone https://github.com/saikiran3944/mle-training.git
cd mle-training
conda env create -f env.yml
cd mle-training
pip install sklearn
python nonstandardcode.py
```
### steps for installing software after you clone the github

```sh
cd mle-training
pip install artifact/mletraining-0.1.12-py3-none-any.whl

# to check if the software installed
pip list
# check for mletraining

# test
python -c 'frpm src.data_ingest import ingest_data'

# also create  processed, saved_models folder
mkdir processed saved_models


```
# Example for test_script running
```python
from src.data_ingest import ingest_data
import pandas as pd
import pickle

from src.model_scoring import score
from src.model_training import train
import os
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(".", "housing")
HOUSING_URL = DOWNLOAD_ROOT + f"datasets/housing/housing.tgz"

ingest_data.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
housing_path = "./housing"
df = ingest_data.load_housing_data(housing_path)
print(df)
ingest_data.data_preprocessing(df,'./processed' )
housing = pd.read_csv('./processed/housing_prepared.csv')
labels = pd.read_csv('./processed/housing_labels.csv')
model_path = './saved_models'
exp_name = 'Dude_the_great'
train.training_with_gridsearchCV(housing,
                                labels,
                                model_path,
                                'dummy',
                                exp_name = exp_name)

imputer = pickle.load(open("./processed/imputer.pkl", "rb"))
model = pickle.load(open(" ./saved_models/linear_regresion.pkl", "rb"))
strat_test_set = pd.read_csv('./processed/strat_test_set.csv')
score.testing_using_strat_test_set(model,
                                    imputer,
                                    strat_test_set)

```

## To check logs
```sh
pwd
# project should be you working directory
```
- check the file with name `custom_config.log`