import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Importing the required libraries.

from sklearn.cluster import KMeans, k_means #For clustering
from sklearn.decomposition import PCA #Linear Dimensionality reduction.
df = pd.read_csv("sales_data_sample.csv") #Loading the dataset.

df.head()

df.shape

df.describe()

df.info()

df.isnull().sum()

df.dtypes

df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS', 'POSTALCODE', 'CITY', 'TERRITORY', 'CONTACTLASTNAME','CONTACTFIRSTNAME', 'STATE', 'PHONE', 'CUSTOMERNAME','ORDERNUMBER']
df = df.drop(df_drop, axis=1) 

df.isnull().sum()


df.dtypes

df[ "COUNTRY" ].unique()

df[ 'PRODUCTLINE'].unique()

df[ 'DEALSIZE'].unique()

productline = pd.get_dummies(df['PRODUCTLINE']) #Converting the categorical co
Dealsize = pd.get_dummies(df['DEALSIZE'])

df = pd.concat([df,productline,Dealsize], axis = 1)

df_drop = ['COUNTRY', 'PRODUCTLINE', 'DEALSIZE'] #Dropping Country too as there
df = df.drop(df_drop, axis=1)

df[ 'PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes #Converting the da

df.drop('ORDERDATE', axis=1, inplace=True)

df.dtypes

distortions = [] # Within Cluster Sum of Squares from the centroid
K = range(1,10)
for k in K:
	kmeanModel = KMeans(n_clusters=k)
	kmeanModel.fit(df)
	distortions.append(kmeanModel.inertia_) #Appeding the intertia to the Di:
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

X_train = df.values #Returns a numpy array.

X_train.shape

model = KMeans(n_clusters=3,random_state=2) #Number of cluster = 3
model = model.fit(X_train) #Fitting the values to create a model.
predictions = model.predict(X_train) #Predicting the cluster values (6,1,or 2)

unique,counts = np.unique(predictions,return_counts=True)

counts = counts.reshape(1,3)
counts_df = pd.DataFrame(counts,columns=['Clusterl', 'Cluster2', 'Cluster3'])

counts_df.head()

pca = PCA(n_components=2) #Converting all the features into 2 columns to make

reduced_X = pd.DataFrame(pca.fit_transform(X_train),columns=['PCA1', 'PCA2']) 

reduced_X.head()

plt.figure(figsize=(14,10))
plt.scatter(reduced, X['PCA1'] , reduced, X['PCA2'])

model.cluster_centers_

reduced_centers = pca.transform(model.cluster_centers_) #Transforming the cent

reduced_centers