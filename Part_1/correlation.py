## @package correlation
#  Module de calcul des correlations.
#  @author Isma
#  @version 1.1.0
#  @date : 217 Octobre 2017

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:02:07 2016

@author: Isma
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
#  @details
#  @param colA   : Première variable
#  @param colB   : Seconde variable
#  @return pivot : Tableau de contingence
#  @date 17 Octobre 2017
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

## Création de la table des fréquences conditionnelles
#  @brief Cette fonction permet de construire à partir d'un tableau de contingence une table des fréquences conditionnelles pour faciliter le calcul du V de Cramer.
#  @details Pour chaque cellule du tableau de contingence, on calucul la fréquence conditionnelle qui lui est associée.
#  @param contingency_tb  : dataframe : Tableau de contingence
#  @return expected_tb    : dataframe : Tableau des fréquences conditionnelles
#  @date : 29 septembre 2017
#  ________________________________________________________________________________________________

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

## Calcul des V de Cramer
#  @brief Cette fonction utilise le tableau de contingence et le tableau des fréquences conditionnelles pour construire le V de Cramer
#  @param contingency_tb  : dataframe : Tableau de contingence
#  @param expected_tb     : dataframe : Tableau des fréquences conditionnelles
#  @return expected_tb    : float : Le V de Cramer entre 2 variables
#  @date : 29 septembre 2017
#  ____________________________________________________________________________________________________


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

## Fonction globale de calcul des V de Cramer
#  @brief Fait appel aux fonctions précédentes contingency_table(), expected_table() et compute_cramer() pour construire la matrice des V de Cramer
#  @details
#  @param colA   : string : Nom de la première variable
#  @param colB   : string : Nom de la seconde variable
#  @return compute_cramer : Matrice des V de Cramer
#  @date : 29 septembre 2017
#  _______________________________________________________________________________________________________________________

def cramer(colA,colB):
    """
    Run contingency_table(), expected_table() and compute_cramer();
    Return Cramer's V from two input columns    
    """
    contingency_tb = contingency_table(colA,colB)
    expected_tb = expected_table(contingency_tb)
    return compute_cramer(contingency_tb, expected_tb)
    

## Fonction globale de calcul des coefficients de correlation
#  @brief Calcul du coefficient de correlation entre 2 variables
#  @details Construction des coefficients de correlation. 
#  @details Ce R² tiendra compte de l'aspect non linéaire de la relation en intégrant des puissances de la variable dans la régrssion.
#  @details On prendra le maximum des coefficients de correlation des formes quadratiques
#  @param X   : array: Feature sur lequel on cherche à calculer le coefficient de correlation avec la VAE à travers la régression
#  @param Y   : array: Variable à Prédire, la VAE
#  @param m   : int : Degres max de la regression : Ce sont les formes quadratiques qui seront intégrées dans la régression.
#  @return R  : float : Maximum des coeffs de regression. On prend la correlation maximale parmi toutes celles qui ressortent de régression avec les formes quadratiques
#  @date : 29 septembre 2017
#  ________________________________________________________________________________________________________


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


## Transformation des variables continues en variable qualitatif de 10 quantiles
#  @brief Transformation des variables continues en variable qualitatif de 10 quantiles
#  @details Lorsqu'une variable est continues et l'autre discrète, la continues est découpée en 10 quantiles.
#  @details Optionnel : Les valeurs manquantes sont mise à zéros
#  @param cont               : string : Nom de la variable continue
#  @param treat_na_as_zero   : boolean: remplacer les valeurs manquante par Zeros. Vaut True par défaut
#  @return cont              : int : variable continue transformée en quantiles
#  @date : 29 septembre 2017
#  _____________________________________________________________________________________________________

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


## Tagging du type de variable
#  @brief Identification des variables continues et des variables discrètes
#  @details Lorsqu'une variable a plus de 12 modalités, on considère qu'elle est continue.
#  @param col     :string : Nom de la variable
#  @param seuil   :int : Nombre maximale de modalité pour considérer qu'une variable est continue
#  @return type_  : string : Un champ type qui indique si la variable est discrète ou continue
#  @date : 29 septembre 2017
#  _______________________________________________________________________________________________________


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

## Calcul de la correlation entre deux variables
#  @brief Calcule la correlation
#  @details Vérifie le type de la variable et appelle la bonne fonction pour calculer soit le R² ou le V de Cramer.
#  @details La fonction renvoie aussi un boolean qui nous renseigne si la correlation est supérieure au seuil défininis.
#  @param colA   :string : Nom de la variable
#  @param colB   :string : Nom du second variable
#  @param seuil   :int : Nombre maximale de modalité pour considérer qu'une variable est continue
#  @param seuil_cramer : float: Seuil minimum de significativité du V de Cramer
#  @param seuil_corr   : float: Seuil minimum de significativité de la correlation
#  @return sortie      : tuple : Renvoie un tuple avec un float et un boolean
#                       - float : la correlation ou le V de Cramer entre deux variables 
#                       - boolean : indique si la correlation est au dessus du seuill qui a été fixé
#  @date : 29 septembre 2017
#
#  _______________________________________________________________________________________________________________________


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


## La construction de la matrice de correlation
#  @brief Construit la matrice de correlation à partir des fonctions adéquates.
#  @details Construit une matrice de correlation et une matrice de boolean qui renseigne si la correlation est suffisante ou pas
#  @details La fonction renvoie aussi  une matrice de boolean qui indique pour chaque cellule si la correlation est supérieure au seuils définis.
#  @param df_ratio     : dataframe : Nom de la table
#  @param seuil        :int : Nombre maximale de modalité pour considérer qu'une variable est continue
#  @param seuil_cramer : float: Seuil minimum de significativité du V de Cramer
#  @param seuil_corr   : float: Seuil minimum de significativité de la correlation
#  @return sortie      : tuple : Renvoie un tuple de deux dataframe
#                       - dataframe :correlation_matrix, la matrice de correlation ou le V de Cramer entre les variables 
#                       - dataframe : correlation_matrix_bool : une matrice de boolean qui indique si la correlation est au dessus du seuill qui a été fixé
#  @date : 29 septembre 2017
#
#  _____________________________________________________________________________________________________________

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


#-------------------------------------------------------------------------
#                           JOIN CORRELATED VARIABLES           
#-------------------------------------------------------------------------

"""
This part of the code return a dictionary grouping the correlated variables
together.
This is done via the recursive function parcourir_colonne()
"""

## Regroupement des variables corellées entre elles
#  @brief Cette fonction renvoie un dictionnaire qui regroupe les variables qui sont corellées entre elles
#  @details C'est une fonction récursive qui s'auto appelle
#  @param col                     : string : Nom de la variable
#  @param colonne_parcourue_arr   : array : colonnes à parcourir
#  @param corel_arr               : array : array des correlations
#  @param dico                    : dictionnaire: Dictionnaire contenant la liste des variables
#  @param key                     : int : Clé de jointure
#  @param column_name             :string : Nom de la colonne
#  @return sortie                 : dictionnaire: Renvoie un dictionnaire qui regrouppe les variables qui sont corellées entre elles
#  @date : 29 septembre 2017
#
#  ___________________________________________________________________________________________________________


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


## Regroupement des variables corellées entre elles
#  @brief Cette fonction renvoie un dictionnaire qui regroupe les variables qui sont corellées entre elles
#  @param colonne_parcourue_arr   : array : colonnes à parcourir
#  @param corel_arr               : array : array des correlations
#  @param dico                    : dictionnaire: Dictionnaire contenant la liste des variables
#  @param column_name             :string : Nom de la colonne
#  @return dico                 : dictionnaire: Renvoie un dictionnaire qui regroupe les variables qui sont corellées entre elles
#  @date : 29 septembre 2017
#  ___________________________________________________________________________________________________________

def fill_dico_by_corr(colonne_parcourue_arr, corel_arr, dico, column_name):
    """
    Returns a dictionary grouping the correlated variables together
    """
    for col in range(0, len(column_name)):
        if colonne_parcourue_arr[col] == False:
            dico[col] = [column_name[col]]
            parcourir_colonne(col, colonne_parcourue_arr, corel_arr, dico, col, column_name)
    return dico


##Regroupement des variables corellées entre elles
#  @brief Cette fonction initialise le dictionnaire et remplis ce dernier avec la fonction fill_dico_by_corr()
#  @param df_ratio     : dataframe : Nom de la table
#  @return dico        : dictionnaire: Renvoie un dictionnaire qui regroupe les variables qui sont corellées entre elless
#  @date : 29 septembre 2017
#  ___________________________________________________________________________________________________________

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

