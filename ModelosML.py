# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:30:30 2021

@author: Ámbar Pérez García
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

############################## KNN ##############################
def knn(X_train, y_train, X_val, y_val, Ks, representation = True):
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))
    
    for n in range(1,Ks):
        #Entrenar el Modelo y Predecir  
        neigh = sk.neighbors.KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
        yhat = neigh.predict(X_val)
        mean_acc[n-1] = sk.metrics.accuracy_score(y_val, yhat)
        std_acc[n-1] = np.std(yhat == y_val)/np.sqrt(yhat.shape[0])
        
    # Representación
    if representation:
        plt.plot(range(1,Ks),mean_acc,'g')
        plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10, color = 'g')
        plt.legend(('Accuracy ', '+/- 3xstd'))
        plt.title('Precisión para los primeros 15 vecinos')
        plt.ylabel('Accuracy ')
        plt.xlabel('Número de Vecinos (K)')
        plt.tight_layout()
        plt.show()
        print("KNN: certeza =", mean_acc.max(), "para k =", mean_acc.argmax()+1)
        neigh = sk.neighbors.KNeighborsClassifier(n_neighbors = mean_acc.argmax()+1).fit(X_train, y_train)
    
    return neigh.predict(X_val)

######################### Decision tree ##########################
def dt(X_train, y_train, X_val, y_val, depth, representation = True):
    mean_acc = np.zeros((depth-1))
    std_acc = np.zeros((depth-1))

    for n in range(1, depth): 
        #Entrenar el Modelo y Predecir 
        Tree = sk.tree.DecisionTreeClassifier(criterion="entropy", max_depth = n)
        Tree.fit(X_train, y_train)
        predTree = Tree.predict(X_val)
        mean_acc[n-1] = sk.metrics.accuracy_score(y_val, predTree)
        
        std_acc[n-1] = np.std(predTree == y_val)/np.sqrt(predTree.shape[0])
        
    # Representación
    if representation:
        plt.plot(range(1, depth), mean_acc,'b')
        plt.fill_between(range(1, depth), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10, color = 'b')
        plt.legend(('Accuracy ', '+/- 3xstd'))
        plt.title('Precisión en función de la profundidad')
        plt.ylabel('Accuracy ')
        plt.xlabel('Profundidad del árbol')
        plt.tight_layout()
        plt.show()
        print("DT: certeza =", mean_acc.max(), "para depth =", mean_acc.argmax()+1) 
        Tree = sk.tree.DecisionTreeClassifier(criterion="entropy", max_depth = mean_acc.argmax()+1)
        Tree.fit(X_train, y_train)

    return Tree.predict(X_val)

############################## SVM ##############################
def svm(X_train, y_train, X_val, y_val, representation = True):
    models = ['linear', 'poly', 'rbf', 'sigmoid']
    mean_acc = np.zeros((len(models)))
    std_acc = np.zeros((len(models)))
    
    for n in range(1, len(models)+1):
        #Entrenar el Modelo y Predecir  
        clf2 = sk.svm.SVC(kernel = models[n-1])
        clf2.fit(X_train, y_train) 
        yhat = clf2.predict(X_val)
        mean_acc[n-1] = sk.metrics.accuracy_score(y_val, yhat)
        
        std_acc[n-1] = np.std(yhat == y_val)/np.sqrt(yhat.shape[0])
        
    # Representación
    if representation:
        plt.plot(range(1,len(models)+1),mean_acc,'k')
        plt.fill_between(range(1,len(models)+1), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10, color = 'k')
        plt.legend(('Accuracy ', '+/- 3xstd'))
        plt.title('Precisión según la función kernel')
        plt.ylabel('Accuracy')
        plt.xlabel('Función kernel')
        plt.xticks(range(1,len(models)+1), ('Linear', 'Polinómica', 'Función de base Radial (RBF)', 'Sigmoide'))
        plt.tight_layout()
        plt.show()
        print( "SVM: certeza =", mean_acc.max(), "para la función kernel", models[mean_acc.argmax()])
        clf2 = sk.svm.SVC(kernel = models[mean_acc.argmax()])
        clf2.fit(X_train, y_train) 
    
    return clf2.predict(X_val)

####################### Logistic regression ########################
def lor(X_train, y_train, X_val, y_val, representation = True):
    models = ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear']
    mean_acc = np.zeros((len(models)))
    std_acc = np.zeros((len(models)))
    
    for n in range(1, len(models)+1):
        
        #Entrenar el Modelo y Predecir  
        LR = sk.linear_model.LogisticRegression(C=0.01, solver=models[n-1]).fit(X_train,y_train)
        yhat = LR.predict(X_val)
        mean_acc[n-1] = sk.metrics.accuracy_score(y_val, yhat)
        
        std_acc[n-1] = np.std(yhat == y_val)/np.sqrt(yhat.shape[0])
        
    # Representación
    if representation:
        plt.plot(range(1,len(models)+1),mean_acc,'r')
        plt.fill_between(range(1,len(models)+1), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10, color = 'r')
        plt.legend(('Accuracy ', '+/- 3xstd'))
        plt.title('Precisión para los distintos parámetros de la regresión logística')
        plt.ylabel('Accuracy')
        plt.xlabel('Parámetros de la regresión')
        plt.xticks(range(1,len(models)+1), ('newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear'))
        plt.tight_layout()
        plt.show()
        print("LoR: certeza =", mean_acc.max(), "para el parámetro de regresión", models[mean_acc.argmax()])
        LR = sk.linear_model.LogisticRegression(C=0.01, solver=models[mean_acc.argmax()]).fit(X_train,y_train) 
    
    return LR.predict(X_val)

####################### Random Forest #####################
def rfc(X_train, y_train, X_val, y_val, n_estimators, representation = True):
    mean_acc = np.zeros((n_estimators-1))
    std_acc = np.zeros((n_estimators-1))
    
    for n in range(1, n_estimators):
        #Entrenar el Modelo y Predecir  
        rf = sk.ensemble.RandomForestClassifier(n_estimators = n).fit(X_train, y_train)
        yhat = rf.predict(X_val)
        mean_acc[n-1] = sk.metrics.accuracy_score(y_val, yhat)
        std_acc[n-1] = np.std(yhat == y_val)/np.sqrt(yhat.shape[0])
        
    # Representación
    if representation:
        plt.plot(range(1,n_estimators),mean_acc,'purple')
        plt.fill_between(range(1,n_estimators), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10, color = 'purple')
        plt.legend(('Accuracy ', '+/- 3xstd'))
        plt.title('Precisión en función de los estimadores')
        plt.ylabel('Accuracy')
        plt.xlabel('Número de estimadores (n)')
        plt.tight_layout()
        plt.show()
        print("RFC: certeza =", mean_acc.max(), "para k =", mean_acc.argmax()+1)
        rf = sk.ensemble.RandomForestClassifier(n_estimators = mean_acc.argmax()+1).fit(X_train, y_train)
    
    return rf.predict(X_val)

####################### Ada Boost #####################
def abc(X_train, y_train, X_val, y_val, n_estimators, representation = True):
    mean_acc = np.zeros((n_estimators-1))
    std_acc = np.zeros((n_estimators-1))
    
    for n in range(1, n_estimators):
        #Entrenar el Modelo y Predecir  
        ab = sk.ensemble.AdaBoostClassifier(n_estimators = n).fit(X_train, y_train)
        yhat = ab.predict(X_val)
        mean_acc[n-1] = sk.metrics.accuracy_score(y_val, yhat)
        std_acc[n-1] = np.std(yhat == y_val)/np.sqrt(yhat.shape[0])
        
    # Representación
    if representation:
        plt.plot(range(1,n_estimators),mean_acc,'pink')
        plt.fill_between(range(1,n_estimators), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10, color = 'pink')
        plt.legend(('Accuracy ', '+/- 3xstd'))
        plt.title('Precisión en función de los estimadores')
        plt.ylabel('Accuracy')
        plt.xlabel('Número de estimadores (n)')
        plt.tight_layout()
        plt.show()
        print("ABC: certeza =", mean_acc.max(), "para k =", mean_acc.argmax()+1)
        ab = sk.ensemble.AdaBoostClassifier(n_estimators = mean_acc.argmax()+1).fit(X_train, y_train)
    
    return ab.predict(X_val)

######################### Confusion Matrix #######################
import itertools
def plot_confusion_matrix(cm, classes, sensor, index, model, normalize=False, cmap=plt.cm.Blues):
    """
    Esta función muestra y dibuja la matriz de confusión.
    La normalización se puede aplicar estableciendo el valor `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = sensor + ' - ' + index + ' - Normalised confusion matrix'
    
    else: 
        title = sensor + ' - ' + index + ' - Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Real label')
    xlab = model + ' predicted label'
    plt.xlabel(xlab)
    plt.show()
