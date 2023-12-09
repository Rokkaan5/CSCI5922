# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# %% working directory
src_file_dir = os.path.dirname(os.path.realpath(__file__)) # directory holding this script file
src_dir = os.path.dirname(src_file_dir)     # parent directory of above directory
os.chdir(src_dir)                           # working directory should now be ".../CSCI5922/Exam3"
print("current working directory:", os.getcwd())

# %% load data
data = pd.read_csv("data/Final_News_DF_Labeled_ExamDataset.csv")
data.head()

# Preprocessing

# %% features vs. targets (aka: estimators vs. predictors, input vs. output, etc.)
X = data.drop(columns=['LABEL']).to_numpy()     # features: everything except column "LABEL"
y = data[["LABEL"]]                             # targets: column "LABEL"

# %% One Hot encode label
OHE = OneHotEncoder()
y = OHE.fit_transform(y).toarray()

# %% Training & Testing set
test_size = 0.2 # what percent of the data = testing set 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=123)

