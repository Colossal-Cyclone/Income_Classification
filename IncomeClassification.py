import pandas as pd

census=pd.read_csv("census_data.csv")

census['income_bracket'].unique()

def labelfix(label):
    if label == ' <=50k':
        return 0
    else:
        return 1

census['income_bracket']=census['income_bracket'].apply(labelfix)

from sklearn.model_selection import train_test_split

x_data=census.drop('income_bracket', axis=1)

y_data=census['income_bracket']

x_train, x_test, y_train, y_test=train_test_split(x_data, y_data, test_size=0.33, random_state=102)

import tensorflow as tf

ag = tf.feature_column.numeric_column("age")
en = tf.feature_column.numeric_column("education_num")
cg = tf.feature_column.numeric_column("capital_gain")
cl = tf.feature_column.numeric_column("capital_loss")
hw = tf.feature_column.numeric_column("hours_per_week")
gr1 = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
oc1 = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
ms1 = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
rp1 = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
ed1 = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
wc1 = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
nc1 = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)
gr=tf.feature_column.embedding_column(gr1,dimension=4)
oc=tf.feature_column.embedding_column(oc1,dimension=4)
ms=tf.feature_column.embedding_column(ms1,dimension=4)
rp=tf.feature_column.embedding_column(rp1,dimension=4)
ed=tf.feature_column.embedding_column(ed1,dimension=4)
wc=tf.feature_column.embedding_column(wc1,dimension=4)
nc=tf.feature_column.embedding_column(nc1,dimension=4)

feat_cols=[ag,en,cg,cl,hw,gr,oc,ms,rp,ed,wc,nc]

input_func=tf.estimator.inputs.pandas_input_fn(x=x_train ,y=y_train ,batch_size=10 ,num_epochs=5000,shuffle=True)

model=tf.estimator.DNNClassifier(hidden_units=[6,6,6,6,6] ,feature_columns=feat_cols,n_classes=2)

model.train(input_fn=input_func,steps=5000)

eval_func=tf.estimator.inputs.pandas_input_fn(x=x_test ,y=y_test ,batch_size=len(x_test) ,shuffle=False)

results=model.evaluate(input_fn=eval_func)

print(results)

