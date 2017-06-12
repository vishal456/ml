import tensorflow as tf
import numpy as np
import pandas as pd

print [[i, 0] for i in range(7)]

columns = ["A"+str(i) for i in range(5)]

df_train = pd.read_csv('data/temp.csv' ,nrows = 4 , names = columns, header = 0)
print df_train["A1"]
print df_train["A1"].values
print df_train["A1"].size