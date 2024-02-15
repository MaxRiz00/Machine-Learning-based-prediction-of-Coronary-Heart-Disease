# Description: This file is used to clean the test dataset.
from helpers import *

def clean_dataset(tx, features_dict):
    """
    Clean the dataset by replacing the missing values with the mean of the column.
    Add new features, square of the _AGEG5YR, _AGEG5YR divided by CVDSTRK3
    Standardize the data
    Add a column of ones to include the bias term in the model

    Args:
        tx: dataset to clean
        features_dict: dictionary of the features, with the name of the feature as key and the index as value

    Returns: cleaned dataset as np.array

    """
    print("cleaning dataset")
    tx = np.array(tx)

    features_indexes = list(features_dict.values())

    # remove the columns that are not in the features_dict
    tx = tx[:, features_indexes]

    def get_index(s):
        return features_indexes.index(features_dict[s])

        # If "PHYSHLTH" , "MENTHLTH" are in features_dict, do the substitution

    # Put to nan the values in columns "PHYSHLTH" , "MENTHLTH" corresponding to 77 ,99

    if (features_dict.get("PHYSHLTH") != None):
        tx[:, get_index("PHYSHLTH")][tx[:, get_index("PHYSHLTH")] == 77] = np.nan
        tx[:, get_index("PHYSHLTH")][tx[:, get_index("PHYSHLTH")] == 99] = np.nan
        tx[:, get_index("PHYSHLTH")][tx[:, get_index("PHYSHLTH")] == 88] = 0

    if (features_dict.get("MENTHLTH") != None):
        tx[:, get_index("MENTHLTH")][tx[:, get_index("MENTHLTH")] == 77] = np.nan
        tx[:, get_index("MENTHLTH")][tx[:, get_index("MENTHLTH")] == 99] = np.nan
        tx[:, get_index("MENTHLTH")][tx[:, get_index("MENTHLTH")] == 88] = 0

    # for column indicated columns put the rows corresponding to 7, 9 to nan
    
    other_columns_9 = np.array(["GENHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1",
                                "CVDSTRK3", "DIABETE3", "SEX", "_RFHLTH", "_RFHYPE5", "_LTASTH1", "_HISPANC", "_RFBMI5",
                                "_EDUCAG", "_INCOMG", "_RFSMOK3", "_TOTINDA", "BLOODCHO", "CHCSCNCR", "CHCOCNCR",
                                "CHCCOPD1", "ADDEPEV2", "CHCKIDNY", "DRDXAR1"])

    for s in other_columns_9:
        if (features_dict.get(s) != None):
            tx[:, get_index(s)][tx[:, get_index(s)] == 9] = np.nan

    other_columns_7 = np.array(["GENHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1",
                                "CVDSTRK3", "DIABETE3", "SEX", "BLOODCHO", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1",
                                "ADDEPEV2", "CHCKIDNY", "DRDXAR1"])

    for s in other_columns_7:
        if (features_dict.get(s) != None):
            tx[:, get_index(s)][tx[:, get_index(s)] == 7] = np.nan
            

    # for "_AGEG5YR", "DROCDY3_" put the rows corresponding to 14, 900 to nan
    
    if (features_dict.get("_AGEG5YR") != None):
        tx[:, get_index("_AGEG5YR")][tx[:, get_index("_AGEG5YR")] == 14] = np.nan
    if (features_dict.get("DROCDY3_") != None):
        tx[:, get_index("DROCDY3_")][tx[:, get_index("DROCDY3_")] == 900] = np.nan

    # make DIABETE3 a binary feature
    if (features_dict.get("DIABETE3") != None):
        tx[:, get_index("DIABETE3")][tx[:, get_index("DIABETE3")] == 2] = 1
        tx[:, get_index("DIABETE3")][tx[:, get_index("DIABETE3")] == 3] = 2
        tx[:, get_index("DIABETE3")][tx[:, get_index("DIABETE3")] == 4] = 2

    # replace nan with the average of the respective feature
    for j in range(tx.shape[1]):
        mean = np.nanmean(tx[:, j])
        for i in range(len(tx[:, j])):
            if tx[i][j] is None or np.isnan(tx[i][j]):
                tx[i][j] = mean

    # add some combination of features to the dataset

    if (features_dict.get("_AGEG5YR") != None):
        tx=np.c_[tx, tx[:, get_index("_AGEG5YR")]**2]
    if (features_dict.get("_AGEG5YR") != None and features_dict.get("CVDSTRK3") != None):
        tx=np.c_[tx, tx[:, get_index("_AGEG5YR")]/tx[:, get_index("CVDSTRK3")]]
    if (features_dict.get("WTKG3") != None and features_dict.get("HTM4") != None):
        tx = np.c_[tx, tx[:, get_index("WTKG3")] / tx[:, get_index("HTM4")]]
    

    #feature standardization 
    def standardize(x):
        centered_data = x - np.mean(x, axis=0)
        std_data = centered_data / np.std(centered_data, axis=0)
        return std_data

    tx = standardize(tx)

    # add a column of ones to include the offset term 
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]

    return tx
