# Cross validation for threshold
from implementations import *
from helpers import *
from clean_dataset import *
import matplotlib.pyplot as plt


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# In[ ]:

DATA_TRAIN_PATH = "data/"

# load data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_TRAIN_PATH)

# replace -1 with 0 in y_train, according to our implementation
y_train = np.where(y_train == -1, 0, 1)

# In[ ]:

### Cross validation ###

features = np.array(["PHYSHLTH", "MENTHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "CVDSTRK3",
                     "DIABETE3", "SEX", "_RFHLTH", "_RFHYPE5", "_LTASTH1",
                     "_AGEG5YR", "_RFBMI5", "_EDUCAG", "_INCOMG", "_RFSMOK3", "DROCDY3_", "_TOTINDA"])
index = [27, 28, 30, 32, 33, 39, 48, 50, 230, 232, 235, 246, 255, 257, 258, 260, 262, 284]

# Build a dictionary with keys = features, values = index
features_dict = dict(zip(features, index))

# Preprocess data
tx_train = clean_dataset(x_train, features_dict)

# split data in k fold
k_fold = 4
seed = 12
k_indices = build_k_indices(y_train, k_fold, seed)


# In[1]:

### Cross validation for threshold###

def cross_validation_th(y, x, k_indices, k, threshold):
    """ cross validation for ridge regression"""

    test_indices = k_indices[k]
    train_indices = k_indices[np.arange(len(k_indices)) != k].flatten()
    _x_train = x[train_indices]
    _x_test = x[test_indices]
    _y_train = y[train_indices]
    _y_test = y[test_indices]

    w_in = np.zeros(x.shape[1])
    w, loss_tr = logistic_regression(_y_train, _x_train, w_in, 1000, gamma=0.1)
    # Compute F1 score and accuracy of test data

    y_pred = np.where(sigmoid(_x_test @ w) < threshold, 0, 1)

    tp = np.sum(y_pred[_y_test == 1] == 1)
    fp = np.sum(y_pred[_y_test == 0] == 1)
    fn = np.sum(y_pred[_y_test == 1] == 0)
    tn = np.sum(y_pred[_y_test == 0] == 0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return f1, accuracy


f1_vector = []
accuracy_vector = []

for th in np.linspace(0.1, 0.5, 5):
    print("Threshold= ", th)
    f1_mean = 0
    accuracy_mean = 0

    for k in range(k_fold):
        f1, accuracy = cross_validation_th(y_train, tx_train, k_indices, k, th)
        f1_mean += f1
        accuracy_mean += accuracy

    accuracy_mean /= k_fold
    f1_mean /= k_fold

    f1_vector.append(f1_mean)
    accuracy_vector.append(accuracy_mean)
    print(" f1_mean = ", f1_mean)
    print(" accuracy_mean = ", accuracy_mean)

    print("")

plt.figure()
plt.plot(np.linspace(0.1, 0.5, 5), accuracy_vector, color='blue', marker='o', label='accuracy')
plt.plot(np.linspace(0.1, 0.5, 5), f1_vector, color='red', marker='x', label='F1 score')

plt.xlabel("Threshold")
plt.ylabel("F1 score, accuracy")
plt.title("F1 score, accuracy vs threshold")
plt.legend(["accuracy", "F1 score"])
plt.savefig("F1_score_accuracy_vs_threshold_010_050.png")

plt.show()


# In[2]:

### Cross validation for lambda ###

def cross_validation_lambda(y, x, k_indices, k, lambda_):
    # cross validation for lamda_ for ridge regression

    test_indices = k_indices[k]
    train_indices = k_indices[np.arange(len(k_indices)) != k].flatten()
    _x_train = x[train_indices]
    _x_test = x[test_indices]
    _y_train = y[train_indices]
    _y_test = y[test_indices]

    w_in = np.zeros(x.shape[1])
    w, loss_tr = reg_logistic_regression(_y_train, _x_train, lambda_, w_in, 1000, gamma=0.1)

    # Compute F1 score and accuracy of test data

    y_pred = np.where(sigmoid(_x_test @ w) < 0.21, 0, 1)

    tp = np.sum(y_pred[_y_test == 1] == 1)
    fp = np.sum(y_pred[_y_test == 0] == 1)
    fn = np.sum(y_pred[_y_test == 1] == 0)
    tn = np.sum(y_pred[_y_test == 0] == 0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return f1, accuracy


f1_vector = []
accuracy_vector = []

for lambda_ in np.logspace(-3, -1, 7):
    print("Lambda= ", lambda_)
    f1_mean = 0
    accuracy_mean = 0

    for k in range(k_fold):
        f1, accuracy = cross_validation_lambda(y_train, tx_train, k_indices, k, lambda_)
        f1_mean += f1
        accuracy_mean += accuracy

    accuracy_mean /= k_fold
    f1_mean /= k_fold

    f1_vector.append(f1_mean)
    accuracy_vector.append(accuracy_mean)
    print(" f1_mean = ", f1_mean)
    print(" accuracy_mean = ", accuracy_mean)

    print("")

# plot for lambda
plt.figure()
plt.semilogx(np.logspace(-3, -1, 7), accuracy_vector, color='blue', marker='o', label='accuracy')
plt.semilogx(np.logspace(-3, -1, 7), f1_vector, color='red', marker='x', label='F1 score')
plt.xlabel("Lambda")
plt.ylabel("F1 score, accuracy")
plt.title("F1 score, accuracy vs lambda")
plt.legend(["accuracy", "F1 score"])
plt.savefig("F1_score_accuracy_vs_lambda_taglio.png")

plt.show()


# In[]


def cross_validation(y, x, k_indices, k):
    test_indices = k_indices[k]
    train_indices = k_indices[np.arange(len(k_indices)) != k].flatten()
    _x_train = x[train_indices]
    _x_test = x[test_indices]
    _y_train = y[train_indices]
    _y_test = y[test_indices]

    w_in = np.zeros(x.shape[1])
    w, loss = reg_logistic_regression(
        _y_train, _x_train, 8e-3, w_in, 1000, gamma=0.1)

    # Compute F1 score and accuracy of test data

    y_pred = np.where(sigmoid(_x_test @ w) < 0.21, 0, 1)

    tp = np.sum(y_pred[_y_test == 1] == 1)
    fp = np.sum(y_pred[_y_test == 0] == 1)
    fn = np.sum(y_pred[_y_test == 1] == 0)
    tn = np.sum(y_pred[_y_test == 0] == 0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return f1, accuracy


# In[3]

### Cross validation removing one variable at a time ###

for i in range(features.shape[0]):
    f1_mean = 0
    accuracy_mean = 0
    print("removed feature=", features[i])
    for k in range(k_fold):
        # Note that tx contains also the offset
        f1, accuracy = cross_validation(
            y_train, tx_train[:, [j for j in range(tx_train.shape[1]) if j != i + 1]], k_indices, k)
        f1_mean += f1
        accuracy_mean += accuracy

    accuracy_mean /= k_fold
    f1_mean /= k_fold

    print(" f1_mean = ", f1_mean)
    print(" accuracy_mean = ", accuracy_mean)

print("")

# In[4]

### Cross validation adding one variable at a time ###
features_to_add = np.array(["GENHLTH", "BLOODCHO", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "ADDEPEV2", "CHCKIDNY",
                            "DRDXAR1", "_HISPANC", "HTM4", "WTKG3"])

index_to_add = [26, 36, 42, 43, 44, 46, 47, 238, 241, 251, 252]

for i in range(features_to_add.shape[0]):
    f1_mean = 0
    accuracy_mean = 0
    print("added feature=", features_to_add[i])
    my_dict = {features_to_add[i]: index_to_add[i]}
    new_feature = clean_dataset(x_train, my_dict)
    for k in range(k_fold):
        # Note that tx contains also the offset
        f1, accuracy = cross_validation(y_train, np.c_[tx_train, new_feature], k_indices, k)
        f1_mean += f1
        accuracy_mean += accuracy

    accuracy_mean /= k_fold
    f1_mean /= k_fold

    print(" f1_mean = ", f1_mean)
    print(" accuracy_mean = ", accuracy_mean)

print("")


