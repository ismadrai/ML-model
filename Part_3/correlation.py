## @package cabp.rfr
#  Module de calcul des corellations
#  @author Ilyas
#  @version 0.1
#  @date 17 Octobre 2017

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:02:07 2016

@author: HyperCube
"""


import pandas as pd
import numpy as np
from scipy import stats
import math
import time
import pdb
from sklearn.linear_model import LinearRegression
lr = LinearRegression() 


## Documentation de la fonction de création des tables de contingences pour le calcul des V de Cramer
#  @brief Cette fonction permet de construire un tableau de contingence pour faciliter le calcul du V de Cramer
#  @date 17 Octobre 2017
#  @details
#  @param colA   : Première variable
#  @param colB   : Second variable
#  @return pivot : Tableau de contingence
#
#  __________________________________________________________________________________________________

#-------------------------------------------------------------------------
#                               CRAMER'S V
#-------------------------------------------------------------------------
"""
For 2 discrete variables, the corelation is computed as cramer's V
See http://www.jybaudot.fr/Inferentielle/associations.html
"""

def contingency_table(colA, colB):
    """
    Create contingency table
    """
    df = pd.DataFrame(columns = ['a','b','c'])
    df['a'] = np.array(colA, dtype=object)
    df['b'] = np.array(colB, dtype=object)
    df['c'] = 1
    pivot = pd.pivot_table(df, values='c', index=['a'], columns=['b'],
       aggfunc=np.sum, fill_value=None, margins=False, dropna=True, margins_name='All')
    pivot.loc['total'] = pivot.sum(axis=0, skipna=True, level=None, numeric_only=None)
    pivot['total'] = pivot.sum(axis=1, skipna=True, level=None, numeric_only=None)
    
    return pivot

## Documentation de la fonction de création de la table des fréquences conditionnelles
#  @brief Cette fonction permet de construire à partir d'un tableau de contingence une table des fréquences conditionnelles pour faciliter le calcul du V de Cramer.
#  @details Pour chaque cellule du tableau de contingence, on calucul la fréquence conditionnelle qui lui est associée.
#  @date 17 Octobre 2017
#  @date 17 Octobre 2017
#  @details
#  @param contingency_tb  : Tableau de contingence
#  @return expected_tb    : Tableau des fréquences conditionnelles
#
#  __________________________________________________________________________________________________

def expected_table(contingency_tb):
    """
    Create expected table
    """
    keys_A = contingency_tb.columns[:-1].tolist()
    keys_B = contingency_tb.index[:-1].tolist()
    expected_tb = pd.DataFrame(data = None, columns = keys_A, index = keys_B)
    
    for i in range(0,len(keys_A)):
        for j in range(0,len(keys_B)):
            expected_tb.iloc[j,i] = float(contingency_tb.iloc[j,-1] * contingency_tb.iloc[-1,i])/float(contingency_tb.iloc[-1,-1])
    
    return expected_tb 

## Documentation de la fonction de calcul des V de Cramer
#  @brief Cette fonction utilise le tableau de contingence et le tableau des fréquences conditionnelles pour construire le V de Cramer
#  
#  @date 17 Octobre 2017
#  @param contingency_tb  : Tableau de contingence
#  @param expected_tb     : Tableau des fréquences conditionnelles
#  @return expected_tb    : Matrice des V de Cramer
#
#  __________________________________________________________________________________________________


def compute_cramer(contingency_tb, expected_tb):
    """
    Return Cramer's V from contingency and expected table
    """
    keys_A = contingency_tb.columns[:-1].tolist()
    keys_B = contingency_tb.index[:-1].tolist()
    cont_tb = contingency_tb.drop('total',axis=1)
    cont_tb = cont_tb.drop('total',axis=0)
    cont_tb[cont_tb.isnull()] = 0
    chi_2 = sum(((cont_tb - expected_tb) * (cont_tb - expected_tb) / expected_tb).sum(axis=1))
    
    l = len(keys_A)
    c = len(keys_B)
    n = contingency_tb.ix[len(keys_B),len(keys_A)]
    chi_2_max = n*(min(l,c)-1)
    
    cramer = math.sqrt(float(chi_2)/float(chi_2_max))
    return cramer

## Documentation de la fonction globale de calcul des V de Cramer
#  @brief Fait appel aux fonctions précédentes pour construire la matrice des V de Cramer
#  
#  @date 17 Octobre 2017
#  @details
#  @param colA   : Première variable
#  @param colB   : Second variable
#  @return compute_cramer : Matrice des V de Cramer
#
#  __________________________________________________________________________________________________

def cramer(colA,colB):
    """
    Run contingency_table(), expected_table() and compute_cramer();
    Return Cramer's V from two input columns    
    """
    contingency_tb = contingency_table(colA,colB)
    expected_tb = expected_table(contingency_tb)
    return compute_cramer(contingency_tb, expected_tb)
    

## Documentation de la fonction globale de calcul des coefficients de corellation
#  @brief Calcul du coefficient de corellation entre 2 variables
#  @details Construction des coefficients de corellation. 
#  @details Ce R² tiendra compte de l'aspect non linéaire de la relation en intégrant des puissances de la variable dans la régrssion.
#  @details On prendra le maximum des coefficients de corellation des formes quadratiques
#  
#  @date 17 Octobre 2017
#  @param X   : Feature 
#  @param Y   : Variable à Prédire
#  @param m   : degres max de la regression : Ce sont les formes quadratiques qui seront intégrées dans la régression.
#  @return R  : max des coeffs de regression. On prend la corellation maximale parmi toutes celles qui ressortent de régression
#
#  __________________________________________________________________________________________________


#-------------------------------------------------------------------------
#               CORRELATION (CONTINUOUS VARIABLES)
#-------------------------------------------------------------------------
"""
For 2 continuous variables, the corelation is computed with pearson method
See https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
"""

def get_R_continuous(X,Y, m = 30):

    Z=pd.DataFrame(index=Y.index)
    R_list = []
    index = ~Y.isnull() & ~X.isnull()
    for i in range(1,m):
        X_i= X ** i
        #X_i.columns = [x for x in X_i.columns+'_'+str(i)]
        Z = pd.concat([Z, X_i], axis=1)
        lr.fit(Z[index], Y[index])
        R_list.append(round(lr.score(Z[index],Y[index]),4))
    R_max = max(R_list)
    return(R_max)


'''
def correlation_using_pandas(colA,colB):
    """
    Return the correlation using pandas
    """
    temp = pd.DataFrame(data = None, columns = ['a','b'])
    temp['a'] = colA
    temp['b'] = colB
    corr = temp.corr(method='pearson', min_periods=1).ix[0,1]
    return corr
'''


## Documentation de transformation des variables continues en variable qualitatif de 10 quantiles
#  @brief Transformation des variables continues en variable qualitatif de 10 quantiles
#  @details Lorsqu'une variable est continues et l'autre discrète, la continues est découpée en 10 quantiles.
#  @details Optionnel : Les valeurs manquantes sont mise à zéros
#  
#  @date 17 Octobre 2017
#  @param cont               : Nom de la variable continue
#  @param treat_na_as_zero   : Variable à Prédire
#  @return R                 : max des coeffs de regression. On prend l
#
#  ___________________________________________________________________________________________________________

#-------------------------------------------------------------------------
#               CORRELATION (CONTINUOUS WITH DISCRETE VARIABLE)
#-------------------------------------------------------------------------
"""
When one variable is discrete and the other is continuous, the continuous
is separated into 10 quantiles
"""



def quantify_col(cont, treat_na_as_zero = True): 
    """
    Takes as input a continuous column and returns the column cutted in
    10 quantiles.
    'Na' are treated as zero if treat_na_as_zero = True; else a 'na' class
    is created
    """
    try:
        quant = cont.quantile(q=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).tolist()
    except:
        pdb.set_trace()
    #pdb.set_trace()
    #Replace na by 0    
    if treat_na_as_zero == True:
        cont[cont.isnull()] = 0    
    cont_copy = cont.copy()
    for i in range(0,len(quant)):
        if i == 0:
            cont[cont_copy <= quant[i]] = str(i+1) 
        elif i == len(quant) -1:
            cont[cont_copy > quant[i]] = str(i+1+1)
            cont[((cont_copy > quant[i - 1]) & (cont_copy <= quant[i]))] = str(i+1)
        else:
            cont[((cont_copy > quant[i - 1]) & (cont_copy <= quant[i]))] = str(i+1)
    
    #Replace na by u'na'
    if treat_na_as_zero == False:
        cont[cont.isnull()] = u'na'
    
    cont.index = range(1,len(cont.index)+1)
    return cont


## Documentation du tagging du type de variable
#  @brief Identification des variables continues et des variables discrètes
#  @details Lorsqu'une variable a plus de 12 modalités, on considère qu'elle est continue.
#  
#  @date 17 Octobre 2017
#  @param col     : Nom de la variable
#  @param seuil   : Nombre maximale de modalité pour considérer qu'une variable est continue
#  @return type_  : Un champ type qui indique si la variable est discrète ou continue
#
#  ___________________________________________________________________________________________________________


#-------------------------------------------------------------------------
#                           CORRELATION MATRIX
#-------------------------------------------------------------------------
"""
This part of the code constructs the correlation matrix
"""

def column_type(col, seuil = 12):
    if len(col.unique()) < seuil or col.dtypes == "object":
        type_ = 'discrete'
    else:
        type_ = 'continuous'
    return type_

## Documentation du calcul de la corellation entre deux variables
#  @brief Calcule la corellation
#  @details Vérifie le type de la variable et appelle la bonne fonction pour calculer soit le R² ou le V de Cramer.
#  @details la fonction renvoie aussi un boolean qui nous renseigne si la corellation est supérieur au seuil défininis.
#  
#  @date 17 Octobre 2017
#  @param colA     : Nom de la variable
#  @param colB   : Nombre maximale de modalité pour considérer qu'une variable est continue
#  @param seuil_cramer     : Nom de la variable
#  @param seuil_corr   : Nombre maximale de modalité pour considérer qu'une variable est continue
#  @return sortie  : Renvoie la corellation entre deux variables et un boolean qui indique si la corellation est au dessus du seuill qui a été fixé

#
#  ___________________________________________________________________________________________________________


def get_correlation(colA, colB, seuil_cramer, seuil_corr):
    """
    Check type of columns and calls the adequate function to compute the 
        correlation
    Returns: (i) the correlation of two columns
            (ii) a boolean representing if the correlation is above a threeshold
    """
    if len(colA.unique()) == 1 or len(colB.unique()) == 1:
        return 0, False
    else:
        if column_type(colA) == 'continuous':
            if column_type(colB) == 'continuous':
                corr = get_R_continuous(colA,colB,m=3)
                return corr, abs(corr) > seuil_corr
            
            else:
                temp = colA.copy()
                temp = quantify_col(temp, treat_na_as_zero = False)
                c = cramer(temp,colB)
                return c, c > seuil_cramer
            
        
        elif column_type(colA) == 'discrete':
            if column_type(colB) == 'continuous':
                temp = colB.copy()
                temp = quantify_col(temp, treat_na_as_zero = False)
                c = cramer(temp,colA)
                return c, c > seuil_cramer
            else:
                c = cramer(colA,colB)
                return c, c > seuil_cramer


## Documentation de la construction de la matrice de corellation
#  @brief Construit la matrice de corellation à partir des fonctions adéquates.
#  @details Construit une matrice de corellation et une matrice de boolean qui renseigne si la correlation est suffisante ou pas
#  @details La fonction renvoie aussi  une matrice de boolean qui indique pour chaque cellule si la corellation est supérieure au seuils définis..
#  
#  @date 17 Octobre 2017
#  @param df_ratio     : Nom de la table
#  @param seuil_cramer : Nombre maximale de modalités pour considérer qu'une variable est continue
#  @param seuil_corr   : Nombre maximales de modalité pour considérer qu'une variable est continue
#  @return sortie      : Renvoie la matrice de corellation entre deux variables et la matrice de boolean qui indique si la corellation est au dessus du seuill qui a été fixé

#
#  ___________________________________________________________________________________________________________

def correlation_matrix(df_ratio, seuil_cramer, seuil_corr):
    """
    Returns two matrices: the correlation matrix, and a matrix of booleans
    indicating if the correlation of two variables is above a certain threeshold
    """
    # Replace missing values by 'na' if columns is object
    for col in df_ratio.dtypes[~((df_ratio.dtypes == 'float') | (df_ratio.dtypes == 'int'))].index:
        df_ratio[col][df_ratio[col].isnull()] = u'na'

    correlation_matrix = pd.DataFrame(data = None, columns = df_ratio.columns,
                                  index = df_ratio.columns)
    correlation_matrix_bool = pd.DataFrame(data = None, columns = df_ratio.columns,
                                      index = df_ratio.columns)
                                   
    for col in range(0,len(correlation_matrix.columns)):
        for index in range(col,len(correlation_matrix.columns)):
            # print('(' + str(col) + ' ' + str(index) + ')')
            if index == col:
                correlation_matrix.ix[col,index] = 1
                correlation_matrix_bool.ix[col,index] = True
            else:
                try:
                    correlation_matrix.ix[col,index], correlation_matrix_bool.ix[col,index] = \
                    get_correlation(df_ratio.ix[:,col], df_ratio.ix[:,index],
                                    seuil_cramer = seuil_cramer, seuil_corr = seuil_corr)
                except ZeroDivisionError:
                    correlation_matrix.ix[col,index] = float('Nan')
                    correlation_matrix_bool.ix[col,index] = False
                correlation_matrix.ix[index,col] = correlation_matrix.ix[col,index]
                correlation_matrix_bool.ix[index,col] = correlation_matrix_bool.ix[col,index]
    
    return correlation_matrix, correlation_matrix_bool



## Documentation du regroupement des variables corellées entre elles
#  @brief Cette fonction renvoie un dictionnaire qui regroupes les variables qui sont corellées entre elles
#  @details C'est une fonction récursive. Elle s'autoappelle
#  
#  @date 17 Octobre 2017
#  @param col     : Nom de la variable
#  @param colonne_parcourue_arr   : colonnes à parcourir
#  @param corel_arr   : array des corellations
#  @param dico   : Dictionnaire contenant la liste des variables
#  @param key   : Clé de jointure
#  @param column_name   : Nom de la colonne
#  @return sortie  : Renvoie un dictionnaire qui regrouppe les variables qui sont corellées entre elles

#
#  ___________________________________________________________________________________________________________

#-------------------------------------------------------------------------
#                           JOIN CORRELATED VARIABLES           
#-------------------------------------------------------------------------

"""
This part of the code return a dictionary grouping the correlated variables
together.
This is done via the recursive function parcourir_colonne()
"""

def parcourir_colonne(col, colonne_parcourue_arr, corel_arr, dico, key, column_name):
    """
    Recursive function
    """
    colonne_parcourue_arr[col] = True
    for ligne in range(0, len(colonne_parcourue_arr)):
        if colonne_parcourue_arr[ligne] == False:
            if corel_arr[ligne,col] == True:
                dico[key].append(column_name[ligne])
                parcourir_colonne(ligne, colonne_parcourue_arr, corel_arr, dico, key, column_name)


def fill_dico_by_corr(colonne_parcourue_arr, corel_arr, dico, column_name):
    """
    Returns a dictionary grouping the correlated variables together
    """
    for col in range(0, len(column_name)):
        if colonne_parcourue_arr[col] == False:
            dico[col] = [column_name[col]]
            parcourir_colonne(col, colonne_parcourue_arr, corel_arr, dico, col, column_name)
    return dico

def join_correlated_variables(df_ratio):
    """
    Do the initilisations and runs fill_dico_by_corr()
    """
    corr_matrix, corr_matrix_bool = correlation_matrix(df_ratio)    
    
    column_name = corr_matrix_bool.columns.values
    corel_arr = corr_matrix_bool.values
    
    dico = {}
    colonne_parcourue_arr = []
    for i in range(0, len(column_name)):
        colonne_parcourue_arr.append(False)
    
    dico = fill_dico_by_corr(colonne_parcourue_arr, corel_arr, dico, column_name)
    return dico

