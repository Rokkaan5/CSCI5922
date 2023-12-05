# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys

# %% working directory
src_file_dir = os.path.dirname(os.path.realpath(__file__)) # directory holding this script file
src_dir = os.path.dirname(src_file_dir)     # parent directory of above directory
os.chdir(src_dir)                           # working directory should now be ".../CSCI5922/Exam3"
print("current working directory:", os.getcwd())

# %% load data
data = pd.read_csv("data/Final_News_DF_Labeled_ExamDataset.csv")
data.head()

# %%
