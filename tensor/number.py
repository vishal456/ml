import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile


columns = ["pixel"+str(i) for i in range(0,784)]
columns1 = ['label'] + columns

test_rows = 41000

df_train = pd.read_csv('data/pixel_train.csv', names = columns1, nrows = test_rows, header = 0 )
df_test = pd.read_csv('data/pixel_train.csv', names = columns1, skiprows = test_rows, nrows = 100, header = 0)


label_col = 'label'


def input_fn(df):
	feature_cols = {k: tf.constant(df[k].values)
                     for k in columns}
	label = tf.constant(df[label_col].values)
	return feature_cols, label

def train_input_fn():
	return input_fn(df_train)

def eval_input_fn():
  	return input_fn(df_test)



features = [tf.contrib.layers.real_valued_column("pixel"+str(i)) for i in range(784)]

model_dir = tempfile.mkdtemp()

m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,feature_columns=features,hidden_units=[100, 50], optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))
m.fit(input_fn=train_input_fn, steps=200)

results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
	print("%s: %s" % (key, results[key]))
