# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 09:50:44 2021

@author: Ámbar Pérez García
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
import ModelosML as ml
import Image2Index as im2in

paths = ["data\AVIRIS_1", "data\AVIRIS_2", "data\MERIS_1", "data\MERIS_2", "data\HICO_1", "data\HICO_2"] 

for path in paths:
    # Cargar imagen y seleccionar píxeles
    indexes, index_name = im2in.Image2Index(path, path[5:-2])

    pixels, classes, Y = im2in.get_labels(path)
    t = im2in.get_values(pixels, indexes)
    models_name = ["KNN", "Decision tree", "Logistic Regression", "Random Forest", "Ada Boost"]

    #Y = np.array(["Water","Water","Water","Thin", "Thin", "Thin","Thick","Thick","Thick","Thick","t","Water","Water","Water","Thin", "Thin", "Thin","Thick","Thick","Thick","Thick","Thin"])
    #X = np.array([[0],[0],[0],[-1], [-1], [-1],[1],[1],[1],[1],[-1],[0],[0],[0],[-1], [-1], [-1],[1],[1],[1],[1],[-1]])
    #classes = ["Water","Thin","Thick"]

    print("Empieza el bucle")

    for i in range(0, len(indexes)):
        print(index_name[i+1])
        X = t[i]
        j = 0 # Reiniciar contador de modelos
        
        # Preprocesado: ecualizar histogramas
        plt.figure()
        im2in.ecualise(X, index_name[0], index_name[i+1], represent= "Stack", Y=Y, classes=classes)
        
        # Dividir el conjunto en train, test y validation
        ###x, x_val, y, y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=3)
        """X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=3)

        # Aplicar modelos
        models = [ml.knn(X_train, y_train, X_test, y_test, Ks=8, representation = False), ml.dt(X_train, y_train, X_test, y_test, depth=7, representation = False), 
                  ml.lor(X_train, y_train, X_test, y_test, representation = False), ml.rfc(X_train, y_train, X_test, y_test, 30, representation = False), 
                  ml.abc(X_train, y_train, X_test, y_test, 30, representation = False)]

        for yhat in models:
            print('\t' + models_name[j])
            cnf_matrix = metrics.confusion_matrix(y_test, yhat, labels=classes)
            ml.plot_confusion_matrix(cnf_matrix, classes=classes, sensor=index_name[0], index=index_name[i+1], model = models_name[j], normalize=True)
            #print(metrics.classification_report(y_test, yhat))
            j += 1"""


    

