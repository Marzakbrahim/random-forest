# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:40:31 2022

@author: HP
"""

##################
# Bibliothèques :#
 #################   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV




#########################
# Importer les données :#
 ########################
donnees = pd.read_excel("C:/Users/HP/Desktop/AssuranceData.xlsx")
donnees.head()    # ou donnees[0:5]
donnees.info()   # pour savoir plusieurs informations sur notre 



##########################################################
# Extraire les attributs avec des valeurs non numériques :#
###########################################################
Type_Dassurance=donnees.values[:,4] # comme ça on aura des arrays dont on peut appliquer les fonctions de numpy
Type_Dassurance=Type_Dassurance.reshape(len(Type_Dassurance),1)

Job=donnees.values[:,5]
Job=Job.reshape(len(Job),1)

Situation_Familiale=donnees.values[:,6]
Situation_Familiale=Situation_Familiale.reshape(len(Situation_Familiale),1)

#print (type(Type_Dassurance))
#print(Type_Dassurance)


###############################################
# Ecodage Binaire des données non numériques :#
###############################################


# 1 : Type_Dassurance
onehot_encoder_Type_Dassurance = OneHotEncoder(sparse=False)    
onehot_encoded_Type_Dassurance = onehot_encoder_Type_Dassurance.fit_transform(Type_Dassurance)
#print(onehot_encoded_Type_Dassurance)


#2 : Job
onehot_encoder_Job = OneHotEncoder(sparse=False)
onehot_encoded_Job = onehot_encoder_Job.fit_transform(Job)
#print(onehot_encoded_Job)


# 3 : Situation_Familiale
onehot_encoder_Situation_Familiale = OneHotEncoder(sparse=False)
onehot_encoded_Situation_Familiale = onehot_encoder_Situation_Familiale.fit_transform(Situation_Familiale)
#print(onehot_encoded_Situation_Familiale)


#######################################################################
# Reconstruire le tableau de données qu'avec des features numériques :#
#######################################################################

Target=donnees.values[:,-1].reshape(len(donnees.values[:,-1]),1)
New_donnees=np.hstack((onehot_encoded_Type_Dassurance,onehot_encoded_Job,onehot_encoded_Situation_Familiale,Target)) # merge des données transférer et la dernière colonne
donneesFinal=np.hstack((donnees.values[:,0:4],New_donnees))  # construire enfin un tableau de donées numériques
donneesFinal[0:5,:]

#######################################################
#Séparation du variable cible à des autres variables :#
#######################################################

features_classes=donneesFinal[:,0:7]
Cible_classe=donneesFinal[:,-1]
#features_classes[0:3]
#Cible_classe[0:15]

#########################################
##### creation de modèle de préduction :#
#########################################
x_train, x_test, y_train, y_test  = train_test_split(features_classes,Cible_classe,test_size=0.25,random_state=42)

modele_rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,ccp_alpha=0.0,max_samples=None,)
modele_rf.fit(x_train, y_train)

