from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def plot_barplots(df):
    for col in df.columns:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(col)
        plt.show()


# loads data
telco = pd.read_csv('telco_2023.csv')

print(telco.head())
print(telco.info())

# scatter plots of columns with float values
df_scat=telco[['longmon','tollmon','equipmon','cardmon','wiremon','churn']]
   
sns.pairplot(df_scat, hue='churn')
plt.savefig('scatterplot.png')

del df_scat

# barplot by churn for the unimportant features
cat_col = ['region', 'marital', 'gender','callid','callwait','custcat']
for x in cat_col:
    sns.countplot(data=telco, x=x, hue='churn', palette='viridis')
    plt.show()
del cat_col

# creating the data df with the important features

data1 = telco.iloc[:, 3:12] 
data2 = telco.iloc[:, 14:17]

data = pd.concat([data1,data2],axis=1) # clustering based on 12 columns

del data1, data2

# barplot by churn for the important features

imp_col = data.columns[5:]

for x in imp_col:
    sns.countplot(data=telco, x=x, hue='churn', palette='viridis')
    plt.show()
del imp_col

# k-means

# elbow method
sse = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

plt.plot(range(1,11), sse, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia/SSE')
plt.show()

#k=3

kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(data)


print('SSE:',kmeans.inertia_)
print('Final locations of the centroid:',kmeans.cluster_centers_)
print("The number of iterations required to converge", kmeans.n_iter_)

#print(kmeans.labels_)

telco['cluster_kmeans'] = kmeans.labels_.tolist()

df_scat=telco[['longmon','tollmon','equipmon','cardmon','wiremon','cluster_kmeans']]

sns.pairplot(df_scat, hue='cluster_kmeans',palette='viridis')
plt.savefig('scatterplot_cluster_kmeans.png')

del df_scat

unique, counts = np.unique(telco['cluster_kmeans'], return_counts=True)
cluster_counts = dict(zip(unique, counts))

print("Number of customers in each cluster:", cluster_counts)

print(telco)
telco.to_csv('telco_cluster_kmeans.csv')


# Agglomerative Clustering

# ward method seems to give better clustering
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data,color_threshold=4)
plt.axhline(y=500, color='black', linestyle='--')
plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=3)
hierarchical_cluster.fit(linkage_data)
labels = hierarchical_cluster.fit_predict(data)

telco['cluster_hier'] = hierarchical_cluster.labels_.tolist()

df_scat=telco[['longmon','tollmon','equipmon','cardmon','wiremon','cluster_hier']]

sns.pairplot(df_scat, hue='cluster_hier',palette='viridis')
plt.savefig('scatterplot_cluster_hier.png')

del df_scat

unique, counts = np.unique(telco['cluster_hier'], return_counts=True)
cluster_counts = dict(zip(unique, counts))

print("Number of customers in each cluster:", cluster_counts)

print(telco)
telco.to_csv('telco_cluster_hier.csv')

# DBSCAN

#search for the eps parameter value
for i in range(2,11):
    neighb = NearestNeighbors(n_neighbors=i) # creating an object of the NearestNeighbors class
    nbrs=neighb.fit(data) # fitting the data to the object
    distances,indices=nbrs.kneighbors(data) # finding the nearest neighbours
    
    # Sort and plot the distances results
    distances = np.sort(distances, axis = 0) # sorting the distances
    distances = distances[:, 1] # taking the second column of the sorted distances
    plt.plot(distances) # plotting the distances
    plt.yticks(np.arange(0, 100, 5))
    plt.show() # showing the plot
    
# the diagram remains the same for every n_neighbors value from 2 to 10
# the elbow occurs at 15 (approximately)

# parameter eps=15
# we test which min_sample gives the best results

for min_samples in range(5,9):
    dbscan = DBSCAN(eps=15, min_samples=min_samples)
    dbscan.fit(data)

    telco['cluster_dbscan'] = dbscan.labels_.tolist()

    df_scat = telco[['longmon','tollmon','equipmon','cardmon','wiremon','cluster_dbscan']]

    sns.pairplot(df_scat, hue='cluster_dbscan',palette='viridis')

    del df_scat
    plt.show();

# by observing the scatter plots min_samples=7 seems to give the best clustering

dbscan = DBSCAN(eps = 15, min_samples = 7)
dbscan.fit(data)

telco['cluster_dbscan'] = dbscan.labels_.tolist()

df_scat = telco[['longmon','tollmon','equipmon','cardmon','wiremon','cluster_dbscan']]

sns.pairplot(df_scat, hue='cluster_dbscan',palette='viridis')
plt.savefig('scatterplot_cluster_dbscan.png')

del df_scat
plt.show();

unique, counts = np.unique(telco['cluster_dbscan'], return_counts=True)
cluster_counts = dict(zip(unique, counts))

print("Number of customers in each cluster:", cluster_counts)

print(telco)
telco.to_csv('telco_cluster_dbscan.csv')


# Which cluster seeems to be the most problematic when it comes to customer churn
pd.crosstab(telco['cluster_kmeans'], telco['churn'], normalize = 'index')
pd.crosstab(telco['cluster_hier'], telco['churn'], normalize = 'index')
pd.crosstab(telco['cluster_dbscan'], telco['churn'], normalize = 'index')

# visualisation of these results

# Plot the count of instances in each cluster
ax = sns.countplot(data=telco, x='cluster_kmeans', hue='churn', palette='viridis')

for i in ax.containers:
    ax.bar_label(i,)
plt.show()

ax = sns.countplot(data=telco, x='cluster_hier', hue='churn', palette='viridis')

for i in ax.containers:
    ax.bar_label(i,)
plt.show()

ax = sns.countplot(data=telco, x='cluster_dbscan', hue='churn', palette='viridis')

for i in ax.containers:
    ax.bar_label(i,)
plt.show()


# Create a new dataframe with the cluster labels and the original features
cluster_data = telco[['cluster_kmeans', 'longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon']]

# Group the data by the cluster labels
grouped = cluster_data.groupby('cluster_kmeans')

# Calculate the mean and standard deviation for each feature in each cluster
cluster_mean = grouped.mean()
cluster_std = grouped.std()

# Print the results
print(cluster_mean)
print(cluster_std)

# Visualize the characteristics of each cluster using box plots
fig, ax = plt.subplots(figsize=(10, 8))
cluster_data.boxplot(by='cluster_kmeans', column=['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon'], ax=ax,grid=False)
ax.set_title('Characteristics of Each Cluster')
ax.set_xlabel('Cluster')
plt.tight_layout()
plt.show()

del cluster_data


# Create a new dataframe with the cluster labels and the original features
cluster_data = telco[['cluster_hier', 'longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon']]

# Group the data by the cluster labels
grouped = cluster_data.groupby('cluster_hier')

# Calculate the mean and standard deviation for each feature in each cluster
cluster_mean = grouped.mean()
cluster_std = grouped.std()

# Print the results
print(cluster_mean)
print(cluster_std)

# Visualize the characteristics of each cluster using box plots
fig, ax = plt.subplots(figsize=(10, 8))
cluster_data.boxplot(by='cluster_hier', column=['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon'], ax=ax,grid=False)
ax.set_title('Characteristics of Each Cluster')
ax.set_xlabel('Cluster')
plt.tight_layout()
plt.show()

del cluster_data

# Create a new dataframe with the cluster labels and the original features
cluster_data = telco[['cluster_dbscan', 'longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon']]

# Group the data by the cluster labels
grouped = cluster_data.groupby('cluster_dbscan')

# Calculate the mean and standard deviation for each feature in each cluster
cluster_mean = grouped.mean()
cluster_std = grouped.std()

# Print the results
print(cluster_mean)
print(cluster_std)

# Visualize the characteristics of each cluster using box plots
fig, ax = plt.subplots(figsize=(10, 8))
cluster_data.boxplot(by='cluster_dbscan', column=['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon'], ax=ax,grid=False)
ax.set_title('Characteristics of Each Cluster')
ax.set_xlabel('Cluster')
plt.tight_layout()
plt.show()

del cluster_data

# analyzing the categorical values of clusters

cluster_data = telco.drop(columns=['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon','cluster_hier','cluster_dbscan'])
cols = cluster_data.columns.tolist()

for i in range(len(cols)):
    sns.barplot(y=cluster_data.iloc[:,i], x='cluster_kmeans', data=cluster_data, palette='viridis')
    plt.show()

    
    