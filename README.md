# Project 1 - ML Course: Machine Learning-based prediction of Coronary Heart Disease

## General info
This project is part of the course CS-433 "Machine Learning" at EPFL. 
The goal of this project is to predict the presence of Coronary Heart Disease (MICHD) in patients using a set of features. 
The train dataset used is taken from the website [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_2015.html): 
it contains 328135 patients and more than 300 features. 

The files `x_train.csv` and `y_train.csv` contain the features and the labels of the train dataset, respectively.
The file `x_test.csv` contains the features of the test dataset.


## Content
* `helpers.py` contains some useful functions to load the data and generate the submission file.
* `implementations.py` contains the implementations of the methods employed to predict the labels.
* `run.py` contains the code to run the project.
* `clean_dataset.py` contains the function used to preprocess the data.
* `cross_validation.py` contains the functions used to perform cross validation.


## Setup
To run this project, install the following libraries:
* [numpy](https://numpy.org/install/)
* [matplotlib](https://matplotlib.org/stable/users/installing.html)

Create a folder `data` and put the file `x_train.csv`, `y_train.csv`, `x_test.csv`  in it.


## Usage

To run the code, with the implementation of our final model, open the terminal and type:
```
python3 run.py
```

This will generate the file `prediction.csv` in the current folder .





