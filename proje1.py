import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)
data = data.rename(columns = {"diagnosis" : "target"})
sns.countplot(data["target"])
print(data.target.value_counts())
data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]
print(len(data))
print(data.head())
data.info()
describe = data.describe()

corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

threshold = 0.75
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("threshold 0.75")

data_melted = pd.melt(data, id_vars = "target",
                              var_name = "features",
                              value_name = "value")

plt.figure()
sns.boxenplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

sns.pairplot(data[corr_features],diag_kind = "kde", markers = "+", hue = "target")
plt.show()    

y = data.target
x = data.drop(["target"],axis = 1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
x_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold

threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()

plt.figure()
plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color = "blue", s = 50, label = "Outliers")
plt.scatter(x.iloc[:,0],x.iloc[:,1],color = "k", s = 3, label = "Data Points")




#Normalization

radius = (x_score.max() - x_score) / (x_score.max() - x_score.min())
plt.scatter(x.iloc[:,0],x.iloc[:,1],color = "k", s = 1000*radius, edgecolors = "r", facecolors = "none", label = "Outlier Scores")
plt.legend()
plt.show()


#Drop outliers

x = x.drop(outlier_index)
y = y.drop(outlier_index).values


#Train, test, split

test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_df = pd.DataFrame(x_train, columns = columns)

x_train_df_describe = x_train_df.describe()
x_train_df["target"] = y_train

data_melted = pd.melt(x_train_df, id_vars = "target",
                              var_name = "features",
                              value_name = "value")

plt.figure()
sns.boxenplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

sns.pairplot(x_train_df[corr_features],diag_kind = "kde", markers = "+", hue = "target")
plt.show()  



#Basic KNN method

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
score = knn.score(x_test,y_test)
print("score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",acc)

#Choose best parameters

def KNN_Best_Params(x_train, x_test,y_train,y_test):
    k_range = list(range(2,31))
    weight_options =["uniform", "distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    y_pred_test = knn.predict(x_test)  
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ", cm_test)
    print("CM Train: ", cm_train)
    
    
    return grid



grid = KNN_Best_Params(x_train, x_test,y_train,y_test)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components = 2)
pca.fit(x_scaled)
x_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(x_reduced_pca, columns = ["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x="p1", y="p2", hue = "target", data = pca_data)
plt.title("PCA: p1 vs p2")

x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_reduced_pca, y, test_size = test_size, random_state = 42)

grid_pca = KNN_Best_Params(x_train_pca, x_test_pca, y_train_pca, y_test_pca)

#Görselleştirme

cmap_light = ListedColormap(['orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','darkblue'])
h=.05 #step size in the mech
x = x_reduced_pca
x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

#put the result into a color plot

z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap = cmap_light)

plt.scatter(x[:,0], x[:,1], c=y, cmap = cmap_bold,edgecolors = 'k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i - Class classification (k = %i, weights = '%s')"%(len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))

#NCA

nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(x_scaled,y)
x_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(x_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1", y = "p2", hue = "target", data = nca_data)
plt.title("NCA: p1 vs p2")

x_train_nca, x_test_nca, y_train_nca, y_test_nca = train_test_split(x_reduced_pca, y, test_size = test_size, random_state = 42)

grid_nca = KNN_Best_Params(x_train_nca, x_test_nca, y_train_nca, y_test_nca)


#Görselleştirme

cmap_light = ListedColormap(['orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','darkblue'])
h=.2 #step size in the mech
x = x_reduced_nca
x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

#put the result into a color plot

z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap = cmap_light)

plt.scatter(x[:,0], x[:,1], c=y, cmap = cmap_bold,edgecolors = 'k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i - Class classification (k = %i, weights = '%s')"%(len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))
