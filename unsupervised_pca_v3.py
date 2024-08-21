import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

#Read Data
data = pd.read_csv('Exasens.csv')
#print(data)

#Deleting Columns by Removing NaN values
updated_data = data
updated_data.dropna(subset = ["ImagMin"], inplace=True)
updated_data.dropna(subset = ["ImagAvg"], inplace=True)
updated_data.dropna(subset = ["RealMin"], inplace=True)
updated_data.dropna(subset = ["RealAvg"], inplace=True)

#Check if they were removed
#print(updated_data)

#LABEL ENCODING
#change diagnosis datatype from object to categorical
updated_data['Diagnosis'] = updated_data['Diagnosis'].astype('category')

# Assigning numerical values and storing in another column
updated_data['Diagnosis_Cat'] = updated_data['Diagnosis'].cat.codes
# COPD = 1
# HC = 2
# Asthma = 0
# Infected = 3

#print(updated_data)
#print(updated_data.Diagnosis_Cat.unique())

#PCA is effected by scale so you need to scale the features in your data before applying PCA.
#Use StandardScaler to help you standardize the dataset’s features onto unit scale (mean = 0 and variance = 1) 
#which is a requirement for the optimal performance of many machine learning algorithms. 
features = ["ImagMin","ImagAvg","RealMin","RealAvg","Gender","Age","Smoking"]

# Separating out the features
x = updated_data.loc[:, features].values

# Separating out the target
y = updated_data.loc[:,['Diagnosis_Cat']].values
y = updated_data.loc[:,['Diagnosis']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

#PCA Projection to 2D
#The original data has 4 columns (sepal length, sepal width, petal length, and petal width).
#The code projects the original data which is 4 dimensional into 2 dimensions. 
#After dimensionality reduction, there usually isn’t a particular meaning assigned to each principal component. 
#The new components are just the two main dimensions of variation.

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

#Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.

finalDf = pd.concat([principalDf, updated_data[['Diagnosis_Cat']]], axis = 1)

#Variability = 60%
print(pca.explained_variance_ratio_)
print(pca.components_)

#Visualize all original dimensions
fig = px.scatter_matrix(
    updated_data,
    dimensions=features,
    color="Diagnosis"
)
fig.update_traces(diagonal_visible=False)
#fig.show()

#Visualize principal components
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    principalComponents,
    labels=labels,
    dimensions=range(2),
    color=updated_data["Diagnosis"]
)
fig.update_traces(diagonal_visible=False)
fig.show()