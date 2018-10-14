import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# prepare data
df = pd.read_csv('inputs_final.csv')
df = df + 1
df[df < 0] = 0
x = np.array([i[0:781] for i in df.values])
y = np.array([i[781] for i in pd.read_csv('inputs_final.csv').values])

# feature selection - determine best features

anova = SelectKBest(f_classif, k=3)
anova.fit(x,y)
print("Best variables:", 
	[df.columns.get_values()[i] for i in anova.get_support(True)])

# model selection, keep 1 and comment rest

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
cs1 = cross_val_score(lda, x, y, cv=5)
print("LDA:", cs1.mean())

# # MLP
# mlp = MLPClassifier(hidden_layer_sizes=(300,150))
# mlp.fit(x,y)
# cs2 = cross_val_score(mlp, x, y, cv=5)
# print("MLP:", cs2.mean())

# # LDA with ANOVA
# pipe = Pipeline([('anova', SelectKBest(f_classif, k=50)),
#                 ('lda', LinearDiscriminantAnalysis())])
# pipe.fit(x,y)
# pipe.score(x,y)
# cs = cross_val_score(pipe, x, y, cv=5).mean()
# print("ANOVA LDA:", cs)

# # MLP with ANOVA
# pipe = Pipeline([('anova', SelectKBest(f_classif, k=50)),
#                 ('mlp', MLPClassifier(hidden_layer_sizes=(10,5)))])
# pipe.fit(x,y)
# pipe.score(x,y)
# cs = cross_val_score(pipe, x, y, cv=5).mean()
# print("ANOVA MLP:", cs)

# # K-Nearest Neighbors with ANOVA
# pipe = Pipeline([('anova', SelectKBest(f_classif, k=15)),
#                 ('knn', KNeighborsClassifier(n_neighbors = 300, 
#                                              weights='distance'))])
# pipe.fit(x,y)
# cs = cross_val_score(pipe, x, y, cv=5).mean()
# print("ANOVA KNN:", cs)