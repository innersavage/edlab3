from sklearn import cluster
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.markers
import matplotlib.lines
import matplotlib.colors
from sklearn.decomposition import PCA
import seaborn
import numpy as np

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

# Zadanie 1 (1 pkt.)
print('\n## W pocie czoła ładuję dane...')
X_train = pandas.read_csv("X_train.txt", sep=' ', header=None)
X_test = pandas.read_csv("X_test.txt", sep=' ', header=None)
y_train = pandas.read_csv("y_train.txt", sep=' ', header=None)
y_test = pandas.read_csv("y_test.txt", sep=' ', header=None)


# Zadanie 2 (2 pkt.)
print('\n## A teraz ciężka praca, tworzymy modele...')
model_svm = svm.SVC(probability=True)
model_svm.fit(X_train, y_train.values.ravel())

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train.values.ravel())

model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(X_train, y_train.values.ravel())

model_random_forest = RandomForestClassifier()
model_random_forest.fit(X_train, y_train.values.ravel())


# Zadanie 3 (1 pkt.)
def counts_from_confusion(confusion):
    counts_list = []
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]
        fn_mask = np.zeros(confusion.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion, fn_mask))
        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion, fp_mask))
        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(confusion, tn_mask))
        counts_list.append({'Class': i,
                            'TP': tp,
                            'FN': fn,
                            'FP': fp,
                            'TN': tn})
    return counts_list


print('\n## Skuteczność klasyfikacji')
for model_name, model in {'SVM': model_svm,
                          'KNN': model_knn,
                          'Decision Tree': model_decision_tree,
                          'Random Forest': model_random_forest}.items():
    print('\n### {}'.format(model_name))
    print('#### Confusion matrix')
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    result = counts_from_confusion(confusion_matrix(y_test, y_pred))
    for cl in result:
        print('##### Klasa {}'.format(cl['Class']))
        for key, value in cl.items():
            print('###### {}: {}'.format(key, value))
    print('#### ACC: {}'.format(accuracy_score(y_test, y_pred)))
    print('#### Recall: {}'.format(recall_score(y_test, y_pred, average='macro')))
    print('#### F1: {}'.format(f1_score(y_test, y_pred, average='macro')))
    print('#### AUC: {}'.format(roc_auc_score(y_test, y_pred_proba, multi_class='ovr')))


# Zadanie 3 (2 pkt.) - czy ktoś zauważył, że dwa razy jest Zadanie 3? ;-)
print('\n## Kros-walidacja')
best = (0, '')
for model_name, model in {'SVM': model_svm,
                          'KNN': model_knn,
                          'Decision Tree': model_decision_tree,
                          'Random Forest': model_random_forest}.items():
    scores = cross_val_score(model, X_test, y_test.values.ravel(), cv=5)
    print('### {}'.format(model_name))
    print('#### Średni wynik: {}'.format(scores.mean()))
    print('#### Odchylenie standardowe: {}'.format(scores.std() * 2))
    if scores.mean() > best[0]:
        best = (scores.mean(), model_name)
print('### Najlepszym algorytmem klasyfikacji wydaje się być: {}'.format(best[1]))


# Zadanie 4 (2 pkt.)
print('\n## A teraz poszukajmy najefektywniejszych parametrów dla algorytmów')
# SVM
print('### SVM')
best = (0, 0)
for i in range(1, 10):
    model_svm = svm.SVC(C=i/10, probability=True)
    model_svm.fit(X_train, y_train.values.ravel())
    scores = cross_val_score(model_svm, X_test, y_test.values.ravel(), cv=5)
    if scores.mean() > best[0]:
        best = (scores.mean(), i/10)
print('#### Dla SVM najefektywniejszą wartością C jest: {}'.format(best[1]))

# KNN
print('### KNN')
best = (0, 0)
for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train.values.ravel())
    scores = cross_val_score(knn, X_test, y_test.values.ravel(), cv=5)
    if scores.mean() > best[0]:
        best = (scores.mean(), k)
print('#### Dla KNN najefektywniejszą ilością sąsiadów jest: {}'.format(best[1]))

# Decision Tree
print('### Decision Tree')
best = (0, 0, 0)
for i in range(1, 10):
    for j in range(1, 10):
        tree = DecisionTreeClassifier(min_samples_split=i/10, min_samples_leaf=j)
        tree.fit(X_train, y_train.values.ravel())
        scores = cross_val_score(tree, X_test, y_test.values.ravel(), cv=5)
        if scores.mean() > best[0]:
            best = (scores.mean(), i/10, j)
print('#### Dla Decision Tree najefektywniejszymi wartościami są'
      ' min_samples_split={} i min_samples_leaf={}'.format(best[1], best[2]))

# Random Forest
print('### Random Forest')
best = (0, 0)
for i in range(1, 50):
    forest = RandomForestClassifier(n_estimators=i)
    forest.fit(X_train, y_train.values.ravel())
    scores = cross_val_score(forest, X_test, y_test.values.ravel(), cv=5)
    if scores.mean() > best[0]:
        best = (scores.mean(), i)
print('#### Dla Random Forest najefektywniejszą wielkością lasu jest: {}'.format(best[1]))


# Zadanie 5 (2 pkt.)
from matplotlib.colors import ListedColormap
print("## Narysujmy trochę wykresów!")
n_classes = 12
h = 0.02
i = 1
pca_2d = PCA(n_components=2)
clf = RandomForestClassifier(n_estimators=best[1])
clf.fit(X_train, y_train.values.ravel())
score = clf.score(X_test, y_test)

ax = plt.subplot(1, 2, 1)

X = pca_2d.fit_transform(X_train)
X_tst = pca_2d.fit_transform(X_test)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

ax.set_title("Input data")
ax.scatter(X[:, 0], X[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
ax.scatter(X_tst[:, 0], X_tst[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

ax = plt.subplot(1, 2, i)

clf = RandomForestClassifier(n_estimators=best[1])
clf.fit(X, y_train.values.ravel())

x_min, x_max = X_tst[:, 0].min() - .5, X_tst[:, 0].max() + .5
y_min, y_max = X_tst[:, 1].min() - .5, X_tst[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
ax.scatter(X[:, 0], X[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
ax.scatter(X_tst[:, 0], X_tst[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Random Forest')
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

plt.tight_layout()
plt.show()
