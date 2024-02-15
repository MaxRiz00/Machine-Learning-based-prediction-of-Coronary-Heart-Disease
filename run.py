import numpy as np
from helpers import *
from implementations import *
from clean_dataset import *

import matplotlib.pyplot as plt
# In[1]
DATA_TRAIN_PATH = "data/"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_TRAIN_PATH)

# replace -1 with 0 in y_train, according to our implementation
y_train = np.where(y_train == -1, 0, 1)

# In[2]

# Preprocess data
# features for the final model

features = np.array(["GENHLTH", "PHYSHLTH", "MENTHLTH", "HLTHPLN1", "CHECKUP1", "BLOODCHO", "CVDSTRK3","CHCSCNCR","CHCOCNCR","CHCCOPD1","ADDEPEV2","CHCKIDNY",
                     "DIABETE3", "SEX", "_RFHYPE5", "_LTASTH1", "DRDXAR1", "_HISPANC",
                     "_AGEG5YR", "HTM4", "WTKG3","_RFBMI5", "_EDUCAG", "_INCOMG", "_RFSMOK3", "DROCDY3_", "_TOTINDA"])

# index of the features in the dataset
index = [26, 27, 28, 30, 33, 36, 39, 42, 43, 44, 46, 47, 48, 50, 232, 235, 238, 241, 246, 251, 252, 255, 257, 258, 260, 262, 284]


"""
# Old features, used for first naive model

features = np.array(["PHYSHLTH", "MENTHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "CVDSTRK3",
                     "DIABETE3", "SEX", "_RFHLTH", "_RFHYPE5", "_LTASTH1",
                     "AGEG5YR", "_RFBMI5", "_EDUCAG", "_INCOMG", "_RFSMOK3", "DROCDY3", "_TOTINDA"])
index = [27, 28, 30, 32, 33, 39, 48, 50, 230, 232, 235, 246, 255, 257, 258, 260, 262, 284]

"""

# Create a dictionary with the features and the corresponding column index
features_dict = {}
for i in range(len(features)):
    features_dict[features[i]] = index[i]


# create tx numpy array from x_train and x_test, with features selected
tx_train = clean_dataset(x_train, features_dict)
tx_test = clean_dataset(x_test, features_dict)

# In[3]
# Logistic regression
print("Logistic regression model:")

# parameters
max_iters = 5000
gamma = 0.1
lambda_ = 8e-3

# initialization of w with zeros
initial_w = np.zeros(tx_train.shape[1])

# run logistic regression
w, loss = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)

# print results
print("w = ", w)
print("loss = ", loss)

# In[4]
#   predict y, with a threshold of 0.25
z = tx_test @ w
probs = sigmoid(z)
y_pred = np.where(probs < 0.25, -1, 1)


# print number of ones
print('number of predicted ones: ', np.sum(y_pred == 1))


# save y
name = "prediction"
create_csv_submission(test_ids, y_pred, name)

