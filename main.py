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
        tree = DecisionTreeClassifier(min_samples_split=i, min_samples_leaf=j)
        tree.fit(X_train, y_train.values.ravel())
        scores = cross_val_score(tree, X_test, y_test.values.ravel(), cv=5)
        if scores.mean() > best[0]:
            best = (scores.mean(), i, j)
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

n_classes = 12
n_trees = 50
plot_step = 0.01

clf = RandomForestClassifier(n_estimators=n_trees)
clf.fit(X_train, y_train.values.ravel())

print("Przygotowania do tworzenia wykresu...")
mrks = list(matplotlib.markers.MarkerStyle.markers.keys())[0:n_classes]
clrs = list(matplotlib.colors.BASE_COLORS.keys())
y_test.columns = ['Class']
y_train.columns = ['Class']
train_dataset = ""
test_dataset = ""
train_dataset = X_train
train_dataset['Class'] = pandas.Series(y_train.values.tolist())
train_dataset['Class'] = [','.join(map(str, l)) for l in train_dataset['Class']]
test_dataset = X_test
test_dataset['Class'] = pandas.Series(y_train.values.tolist())
test_dataset['Class'] = [','.join(map(str, l)) for l in test_dataset['Class']]
train_dataset['Class'] = pandas.to_numeric(train_dataset['Class'])
test_dataset['Class'] = pandas.to_numeric(test_dataset['Class'])

x_min, x_max = float(X_train.iloc[:, 0].min()) - 1, float(X_train.iloc[:, 0].max()) + 1
y_min, y_max = float(X_train.iloc[:, 1].min()) - 1, float(X_train.iloc[:, 1].max()) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig, axes = plt.subplots(nrows=3, ncols=1, sharex='True', sharey='False', figsize=(12, 12))
axes[1].contourf(xx, yy, Z)

x_min, x_max = float(X_test.iloc[:, 0].min()) - 1, float(X_test.iloc[:, 0].max()) + 1
y_min, y_max = float(X_test.iloc[:, 1].min()) - 1, float(X_test.iloc[:, 1].max()) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[2].contourf(xx, yy, Z)

for i in range(1, n_classes + 1):
    sliced_train = train_dataset.loc[train_dataset.Class == i]
    sliced_test = test_dataset.loc[test_dataset.Class == i]
    sliced_train.plot(x=0, y=1, ax=axes[0], kind='scatter', marker=mrks[i-1], c=clrs[i-1], label="Klasa {}".format(i))
    sliced_test.plot(x=0, y=1, ax=axes[0], kind='scatter', marker=mrks[i-1], c=clrs[i-1])
    sliced_train.plot(x=0, y=1, ax=axes[1], kind='scatter', marker=mrks[i-1], c=clrs[i-1], label="Klasa {}".format(i), zorder=1)
    sliced_test.plot(x=0, y=1, ax=axes[2], kind='scatter', marker=mrks[i-1], c=clrs[i-1], label="Klasa {}".format(i), zorder=1)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

