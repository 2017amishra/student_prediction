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
import warnings
warnings.filterwarnings("ignore")

# prepare data
df = pd.read_csv('inputs_final.csv')
df = df + 1
df[df < 0] = 0
x = np.array([i[0:781] for i in df.values])
y = np.array([i[781] for i in pd.read_csv('inputs_final.csv').values])

# feature selection - determine best features
x_orig = x
y_orig = y
anova = SelectKBest(f_classif, k=3)
anova.fit(x_orig, y_orig)
print("Best variables:", 
	[df.columns.get_values()[i] for i in anova.get_support(True)])

# model selection

# LDA
x_orig = x
y_orig = y
lda = LinearDiscriminantAnalysis()
lda.fit(x_orig, y_orig)
cs1 = cross_val_score(lda, x_orig, y_orig, cv=5)
print("LDA:", cs1.mean())

# MLP
x_orig = x
y_orig = y
mlp = MLPClassifier(hidden_layer_sizes=(300,150))
mlp.fit(x_orig, y_orig)
cs2 = cross_val_score(mlp, x_orig, y_orig, cv=5)
print("MLP:", cs2.mean())

# LDA with ANOVA
x_orig = x
y_orig = y
pipe = Pipeline([('anova', SelectKBest(f_classif, k=50)),
                ('lda', LinearDiscriminantAnalysis())])
pipe.fit(x_orig, y_orig)
pipe.score(x_orig, y_orig)
cs = cross_val_score(pipe, x_orig, y_orig, cv=5).mean()
print("ANOVA LDA:", cs)

# MLP with ANOVA
x_orig = x
y_orig = y
pipe = Pipeline([('anova', SelectKBest(f_classif, k=50)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(10,5)))])
pipe.fit(x_orig, y_orig)
pipe.score(x_orig, y_orig)
cs = cross_val_score(pipe, x_orig, y_orig, cv=5).mean()
print("ANOVA MLP:", cs)

# K-Nearest Neighbors with ANOVA
x_orig = x
y_orig = y
pipe = Pipeline([('anova', SelectKBest(f_classif, k=15)),
                ('knn', KNeighborsClassifier(n_neighbors = 300, 
                                             weights='distance'))])
pipe.fit(x_orig, y_orig)
cs = cross_val_score(pipe, x_orig, y_orig, cv=5).mean()
print("ANOVA KNN:", cs)