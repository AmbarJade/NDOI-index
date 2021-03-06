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
import pandas as pd

paths = ["data\AVIRIS_1", "data\AVIRIS_2", "data\MERIS_1", "data\MERIS_2", "data\HICO_1", "data\HICO_2"] 
df = pd.DataFrame()

titles = ["AVIRIS - 17 May 2010", "AVIRIS - 18 May 2010", "MERIS - 26 April 2010", "MERIS - 2 May 2010", "HICO - 24 May 2010", "HICO - 28 May 2010", ]
titles = ["AVIRIS 1", "AVIRIS 2", "MERIS 1", "MERIS 2", "HICO 1", "HICO 2", ]

for path in paths:
    # Cargar imagen y seleccionar píxeles
    sensor = path[5:-2]
    indexes, index_name = im2in.Image2Index(path, sensor)

    pixels, classes, Y = im2in.get_labels(path, ".json")
    t = im2in.get_values(pixels, indexes)
    models_name = ["KNN", "Decision tree", "Random Forest", "Ada Boost"]

    print("Empieza el bucle")
    for i in range(0, len(indexes)):
        X = t[i]
        j = 0 # Reiniciar contador de modelos
        
        # Preprocesado: ecualizar histogramas
        plt.figure()
        im2in.histogram(X, sensor, index_name[i], represent= "Stack", Y=Y, classes=classes)
        
        # Dividir el conjunto en train, test y validation
        ###x, x_val, y, y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=3)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=3)

        # Aplicar modelos
        """
        j = 0 # Reiniciar contador de modelos
        models = ml.knn(X_train, y_train, X_test, y_test, Ks=8, representation = False)#[ml.knn(X_train, y_train, X_test, y_test, Ks=8, representation = False), ml.dt(X_train, y_train, X_test, y_test, depth=7, representation = False), 
                  #ml.rfc(X_train, y_train, X_test, y_test, 30, representation = False),  ml.abc(X_train, y_train, X_test, y_test, 30, representation = False)]

        for yhat in models:
            #print('\t' + models_name[j])
            plt.figure()
            # Calcular y representar matrices de confusión
            cnf_matrix = metrics.confusion_matrix(y_test, yhat, labels=classes)
            ml.plot_confusion_matrix(cnf_matrix, classes=classes, sensor=sensor, index=index_name[i], model = models_name[j], normalize=True)
            
            # Tabla de pandas para visualizar errores
            errors = metrics.classification_report(y_test, yhat, output_dict=True)
            df = ml.error_df(path[5:], index_name[i], models_name[j], classes, errors, df)
            j += 1
        """
        # Aplicar un único modelo
        yhat = ml.knn(X_train, y_train, X_test, y_test, Ks=8, representation = False)

        plt.figure()
        # Calcular y representar matrices de confusión
        cnf_matrix = metrics.confusion_matrix(y_test, yhat, labels=classes)
        ml.plot_confusion_matrix(cnf_matrix, classes=classes, sensor=sensor, index=index_name[i], model = models_name[0], normalize=True)
        
        # Tabla de pandas para visualizar errores
        errors = metrics.classification_report(y_test, yhat, output_dict=True)
        df = ml.error_df(path[5:], index_name[i], models_name[0], classes, errors, df)
            
#df.to_excel('data\df_2classes.xlsx')

    

