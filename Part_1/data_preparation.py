
## @package data_preparation
#  Module de la data préparation.
#  @author Isma
#  @version 1.1.0
#  @date 29 septembre 2017

import pandas as pd
import numpy as np
import glob, os.path
from math import *
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import correlation as corr
import re
lr = LinearRegression() 


## Chargement des données
#  @brief Cette fonction permet de charger les données explicatives
#  @date 17 Octobre 2017
#  @details Elle supprime les colonnes ayant les mêmes valeurs sur chaque ligne, reindexe les données
#  @param fichier   : Pnom ou chemin du fichier csv contenant les données explicatives
#  @param taille_ech   : nombre de lignes à charger (par défaut charge toutes les lignes)
#  @param drop   : liste de colonnes à supprimer
#  @param index_col    : nom de la colonne à utiliser pour indexer les données
#  @return T : Dataframe des données chargées
#
#  __________________________________________________________________________________________________
def charge_expl(fichier = 'matrice_ilb_201603.csv', taille_ech = None, drop = [], index_col = None):
    
    # Charge données
    #T = pd.read_csv(fichier, nrows = taille_ech, sep=';')
    T = fichier
    # Supprime les colonnes non-voulues
    T.drop(drop, axis = 1, inplace = True)
    # Reindexe
    if index_col != None:
        T.index = T[index_col]
        T.drop(index_col, axis = 1, inplace = True)
    # Supprime les colonnes inutiles, i.e. ayant la meme valeur sur toutes les
    # lignes
    for col in T.columns:
        if len(T[col].unique()) == 1:
            if sum(T[col].isnull()) == 0:
                T.drop(col, axis = 1, inplace = True)
    return T


## Charge les données à expliquer
#  @brief Cette fonction permet de charger les données à expliquer
#  @date 17 Octobre 2017
#  @details Elle permet de charger les données à expliquer
#  @param T   : dataframe des varaibles explicatives
#  @param fichier_sortie: nom ou chemin du fichier csv contenant la variable à expliquer
#  @param input_cal: nom de la colonne contenant la variable de sortie 
#  @param index_col: nom de la colonne à utiliser pour indexer les données
#  @return T : serie de la variable à expliquer
#
#  __________________________________________________________________________________________________
def charge_sortie(T, fichier_sortie, input_col, index_col = None):

    # Charge données
    #temp = pd.read_csv(fichier_sortie, sep=';', encoding='utf-8')
    temp = fichier_sortie
    # Reindexe
    if index_col != None:
        temp.index = temp[index_col]
        temp.drop(index_col, axis = 1, inplace = True)
    #enlever element aberrante
    temp = temp[temp[input_col] != 1]
    
    Y = pd.DataFrame(index = T.index)
    Y.loc[temp.index.intersection(Y.index),'sortie'] = temp.loc[temp.index.intersection(Y.index), input_col]
    return Y['sortie']

## Sélectionne les colonnes discrètes et quantitatives
#  @brief Cette fonction de selectionner les colonnes discrètes et les colonnes quantitatives
#  @date 17 Octobre 2017
#  @details Elle crée deux listes : l'une contenant les noms des colonnes discrètes et l'autre les noms des colonnes continues
#  @param T   : dataframe des varaibles explicatives
#  @return discrete_column: liste des colonnes discretes
#  @return quantitatives_column: liste des colonnes quantitatives
#
#  __________________________________________________________________________________________________

def get_column_type(T, seuil = 12):

    discrete_column = []
    quantitative_column = []
    for col in T.columns:
        if len(T[col].unique()) < seuil or T.dtypes[col] == "object":
            discrete_column.append(col)
        else:
            quantitative_column.append(col)
    return discrete_column, quantitative_column


# =============================================================================
#                         Traite colonnes discretes
# =============================================================================

## Traitement des colonnes discrètes
#  @brief Cette fonction traite les colonnes discrètes en identifiant les variables assez corrélés avec le Y à retenir dans la suite de l'algorithme
#  @details Elle crée deux sortie : Un dataframe transformé et un dictionnaire regroupant les corellation des variables
#  @param T        : dataframe : Dataframe des variables explicatives discretes
#  @param Y        : dataframe : Dataframe de la sortie
#  @param R_dico   : dictionnaire: Dictionnaire regroupant les R des variables explicatives
#  @param  verbose : boolean : Afficher ou pas l'avancee de la fonction
#  @param method   : string : Méthode utilisée pour calculer la corrélation entre la variable explivative discrète et la VAE. Prend que les valeurs suivantes :
#                         - regression si la variable explicative est continue
#                         -  Cramer si la variable explicative est discrète
#  @param R_min    : float : Valeur minimale de la corellation
#  @return F, R_dico: tuple : Renvoie un tuple contenant les objets suivants :
#                            - dataframe transformé
#                            - dictionnaire regroupant les R des variables explicatives
#  @date 29 septembre 2017
#  ____________________________________________________________________________________________________________________________________________________

def treat_discrete_columns(T, Y, R_dico, dic, method = 'regression', R_min = 0.1, verbose = False):

    if verbose == True:
        def vprint(*args):
            # Print each argument separately so caller doesn't need to
            # stuff everything to be printed into a single string
            for arg in args:
               print(arg,)
            print
    else:   
        vprint = lambda *a: None      # do-nothing function    
    
    # Replace nan
    for col in T.columns:
        if T.dtypes[col]=='object' or T.dtypes[col]=='O':
                T.loc[T[col] == ".",col] = 'Na'
        T.loc[T[col].isnull(),col] = float('NaN')
      
    F = pd.DataFrame(index = T.index)
    drop_tb = pd.DataFrame(columns=['col_name','R2'])
    drop_index = 0
    keep_tb = pd.DataFrame(columns=['col_name','R2'])
    keep_index = 0
    keep_tb_1 = pd.DataFrame(columns=['col_name','R2'])
    keep_index_1 = 0
    
    if method == 'regression':
        for col in T.columns:
            if (col in dic)== True:
                # Construit TDC
                T[col] = T[col].astype(str)
                T[col] = T[col].astype('category')
                tdc = pd.DataFrame(pd.get_dummies(T[col]))
                new_col_name =[]
                for i in range(0,len(tdc.columns)):
                    new_col_name.append(col + '_' + str(tdc.columns[i]))
                tdc.columns = new_col_name
                # Garde ou pas les variables du TDC
                index = ~Y.isnull()
                for i in tdc.columns:
                    if len(tdc.loc[index,i].unique()) == 2:
                        R=dic[col][1]
                        vprint('keep ' + i + ' with R2 = ' + str(round(R,3)))
                        keep_tb_1.loc[keep_index_1] = [i,R]
                        keep_index_1 = keep_index_1 + 1 
                    
            else:    
                # Construit TDC
                T[col] = T[col].astype(str)
                T[col] = T[col].astype('category')
                tdc = pd.DataFrame(pd.get_dummies(T[col]))
                new_col_name =[]
                for i in range(0,len(tdc.columns)):
                    new_col_name.append(col + '_' + str(tdc.columns[i]))
                tdc.columns = new_col_name
                # Garde ou pas les variables du TDC
                index = ~Y.isnull()
                for i in tdc.columns:
                    R, bool_R = corr.get_correlation(tdc.loc[index,i], Y.loc[index],
                                            seuil_cramer = 1, seuil_corr = 1)
                    if len(tdc.loc[index,i].unique()) == 2:
                        R, bool_R = corr.get_correlation(tdc.loc[index,i], Y.loc[index],
                                            seuil_cramer = 1, seuil_corr = 1)                                       
                        if R < R_min:
                            vprint('vire ' + i + ' with R2 = ' + str(round(R,3)))
                            tdc.drop(i, axis = 1, inplace = True)
                            drop_tb.loc[drop_index] = [i,R]
                            drop_index = drop_index + 1
                        else:
                            vprint('keep ' + i + ' with R2 = ' + str(round(R,3)))
                            keep_tb.loc[keep_index] = [i,R]
                            keep_index = keep_index + 1
                    else:                    
                        vprint('vire ' + i + ' with R2 = ' + str(round(R,3)))
                        tdc.drop(i, axis = 1, inplace = True)
                        drop_tb.loc[drop_index] = [i,R]
                        drop_index = drop_index + 1
            # Add tdc to F
            F = pd.concat([F,tdc], axis = 1)
            del(tdc)
        
        drop_tb.sort_values(by = 'R2', inplace = True, ascending = False)
        keep_tb.sort_values(by = 'R2', inplace = True, ascending = False)
        R_dico['variables'] = pd.concat([R_dico['variables'],keep_tb_1], axis = 0)
        R_dico['variables discretes gardees'] = pd.concat([R_dico['variables discretes gardees'],keep_tb], axis = 0)
        R_dico['variables discretes jetees'] = pd.concat([R_dico['variables discretes jetees'],drop_tb], axis = 0)
        return F, R_dico
        
    elif method == 'Cramer':
        for col in T.columns:
            index = T[col].apply(np.isreal)
            index = index & ~T[col].isnull()
            T[col][index] = T[col][index].astype(int)
            T[col] = T[col].astype(str)            
            # Cramer
            index = ~Y.isnull()
            if len(T.loc[index,col].unique()) > 1:
                R, bool_R =  corr.get_correlation(T.loc[index,col], Y.loc[index], 1, 1)
                if (col in dic)== True:
                    R=dic[col][1]
                    vprint('garde ' + col + ' avec R2 = ' + str(round(R,3)))
                    keep_tb_1.loc[keep_index_1] = [col,R]
                    keep_index_1 = keep_index_1 + 1
                    replace_dico = {}
                    for value in T[col].unique():
                        index = (T[col] == value)
                        replace_dico[value] = round(Y.loc[index].mean(),0)
                    F = pd.concat([F,T[col].replace(replace_dico)], axis =1)
                elif R < R_min:
                    vprint('vire ' + col + ' avec R2 = ' + str(round(R,3)))
                    drop_tb.loc[drop_index] = [col,R]
                    drop_index = drop_index + 1
                else:
                    vprint('garde ' + col + ' avec R2 = ' + str(round(R,3)))
                    keep_tb.loc[keep_index] = [col,R]
                    keep_index = keep_index + 1
                    replace_dico = {}
                    for value in T[col].unique():
                        index = (T[col] == value)
                        replace_dico[value] = round(Y.loc[index].mean(),0)
                    F = pd.concat([F,T[col].replace(replace_dico)], axis =1)
                
            else:
                vprint('vire ' + col + ' car valeurs constantes')
                drop_tb.loc[drop_index] = [col,0]
                drop_index = drop_index + 1
    
        drop_tb.sort_values(by = 'R2', inplace = True, ascending = False)
        keep_tb.sort_values(by = 'R2', inplace = True, ascending = False)
        R_dico['variables'] = pd.concat([R_dico['variables'],keep_tb_1], axis = 0)
        R_dico['variables discretes gardees'] = pd.concat([R_dico['variables discretes gardees'],keep_tb], axis = 0)
        R_dico['variables discretes jetees'] = pd.concat([R_dico['variables discretes jetees'],drop_tb], axis = 0)
        return F, R_dico
    
    else:
        raise ValueError('methode non reconnue')


# =============================================================================
#                         Traite colonnes continues
# =============================================================================

## Traitement des colonnes continues
#  @brief Cette fonction traite les colonnes continues en identifiant les variables assez corrélées avec le Y à retenir dans la suite de l'algorithme.
#  @details Elle crée deux sorties : Un dataframe transformé et un dictionnaire regroupant les corellations des variables
#  @param E: dataframe : dataframe des variables explicatives quantitatives
#  @param Y: dataframe : dataframe de la sortie
#  @param R_dico: dictionnaire: dictionnaire regroupant les R des variables explicatives
#  @param  verbose: boolean : afficher ou pas l'avancee de la fonction
#  @param method   : string : Méthode utilisée pour calculer la corrélation entre la variable explivative discrète et la VAE. Prend que les valeurs suivantes :
#                         - regression si la variable explicative est continue
#                         -  Cramer si la variable explicative est discrète
#  @param R_min : float :  seuil minimun à partir desquels les variables sont conservées
#  @param R_cont_y: float : R minimum pour les variables continues
#  @param R_Cramer_y: float : R minimum pour les variables discretes
#  @return E: dataframe : dataframe transformé
#  @return R_dico: dictionnaire: dictionnaire regroupant les R des variables explicatives
#  @date 29 septembre 2017
#  __________________________________________________________________________________________________


def treat_continuous_columns(E, Y, R_dico, dic, method = 'regression',R_min = 0.1, R_cont_y = 0.3,
                             R_Cramer_y = 0.25, verbose = False):

    if verbose == True:
        def vprint(*args):
            # Print each argument separately so caller doesn't need to
            # stuff everything to be printed into a single string
            for arg in args:
               print(arg,)
            print
    else:   
        vprint = lambda *a: None      # do-nothing function    
        
    G = pd.DataFrame(index = E.index)    
    drop_tb = pd.DataFrame(columns=['col_name','R2'])
    drop_index = 0
    keep_tb = pd.DataFrame(columns=['col_name','R2'])
    keep_index = 0
    #
    drop_tb_q = pd.DataFrame(columns=['col_name','R2'])
    drop_q_index = 0
    keep_tb_d = pd.DataFrame(columns=['col_name','R2'])
    keep_d_index = 0
    keep_tb_q = pd.DataFrame(columns=['col_name','R2'])
    keep_q_index = 0
    #
    keep_tb_d_1 = pd.DataFrame(columns=['col_name','R2'])
    keep_d_index_1 = 0
    keep_tb_q_1 = pd.DataFrame(columns=['col_name','R2'])
    keep_q_index_1 = 0
    
    if method == 'regression':
        for col in E.columns:
            if (col in dic) == True:
                # Add a log variable
                index = (E[col] < (E[col].mean() - 4*E[col].std())) | \
                        (E[col] > (E[col].mean() + 4*E[col].std()))
                if sum(index) > 0:
                    sgn = (E[col] - E[col].mean()) / abs(E[col] - E[col].mean())
                    v = pd.DataFrame(data = 0, columns = ['log_' + col] , index = E.index)
                    v.loc[index] = sgn.loc[index] * np.log(sgn.loc[index] * (E.loc[index, col] - E.loc[:,col].mean()) \
                                    / (4 *E.loc[:,col].std()))
                    # Garde ou vire la log variable
                    R = dic[col][1]
                    vprint('garde log_' + col + ' avec R2 = ' + str(round(R,3)))
                    keep_tb_q_1.loc[keep_q_index_1] = ['log_' + col,R]
                    keep_q_index_1 = keep_q_index_1 + 1
                    E = pd.concat([E,v], axis = 1)
                
                # Add a nan variable           
                index = E[col].isnull()
                if sum(index) > 0:             
                    u = pd.DataFrame(data = 0, columns = ['nan_'+col] , index = E.index)
                    u.loc[index] = 1
                    # Garde ou pas la variable nan
                    index_ = ~Y.isnull()
                    R = dic[col][1]
                    vprint('garde nan_' + col + ' avec R2 = ' + str(round(R,3)))
                    keep_tb_d_1.loc[keep_d_index_1] = ['nan_' + col,R]
                    keep_d_index_1 = keep_d_index_1 + 1
                    E = pd.concat([E,u], axis = 1)
                
                # Garde ou vire la variable
                R = dic[col][1]
                vprint('garde ' + col + ' with R2 = ' + str(round(R,3)))
                keep_tb_q_1.loc[keep_q_index_1] = [col,R]
                keep_q_index_1 = keep_q_index_1 + 1
            else:                
                # Add a log variable
                index = (E[col] < (E[col].mean() - 4*E[col].std())) | \
                        (E[col] > (E[col].mean() + 4*E[col].std()))
                if sum(index) > 0:
                    sgn = (E[col] - E[col].mean()) / abs(E[col] - E[col].mean())
                    v = pd.DataFrame(data = 0, columns = ['log_' + col] , index = E.index)
                    v.loc[index] = sgn.loc[index] * np.log(sgn.loc[index] * (E.loc[index, col] - E.loc[:,col].mean()) \
                                    / (4 *E.loc[:,col].std()))
                    # Garde ou vire la log variable
                    R = corr.get_R_continuous(v.iloc[:,0],Y, m = 3)
                    if R > R_cont_y:
                        vprint('garde log_' + col + ' avec R2 = ' + str(round(R,3)))
                        keep_tb_q.loc[keep_q_index] = ['log_' + col,R]
                        keep_q_index = keep_q_index + 1
                        E = pd.concat([E,v], axis = 1)
                
                # Add a nan variable           
                index = E[col].isnull()
                if sum(index) > 0:             
                    u = pd.DataFrame(data = 0, columns = ['nan_'+col] , index = E.index)
                    u.loc[index] = 1
                    # Garde ou pas la variable nan
                    index_ = ~Y.isnull()
                    R, bool_R =  corr.get_correlation(u.loc[index_,'nan_'+col], Y.loc[index_],
                                                seuil_cramer = 1, seuil_corr = 1)
                    if R > R_Cramer_y:
                        vprint('garde nan_' + col + ' avec R2 = ' + str(round(R,3)))
                        keep_tb_d.loc[keep_d_index] = ['nan_' + col,R]
                        keep_d_index = keep_d_index + 1
                        E = pd.concat([E,u], axis = 1)
                
                # Garde ou vire la variable
                R = corr.get_R_continuous(E[col],Y,3)
                if R > R_cont_y:
                    vprint('garde ' + col + ' with R2 = ' + str(round(R,3)))
                    keep_tb_q.loc[keep_q_index] = [col,R]
                    keep_q_index = keep_q_index + 1
                else:
                    vprint('vire ' + col + ' with R2 = ' + str(round(R,3)))
                    E.drop(col, axis = 1, inplace = True)
                    drop_tb_q.loc[drop_q_index] = [col,R]
                    drop_q_index = drop_q_index + 1
        R_dico['variables'] = pd.concat([R_dico['variables'],keep_tb_q_1], axis = 0)
        R_dico['variables'] = pd.concat([R_dico['variables'],keep_tb_d_1], axis = 0)
        R_dico['variables continues gardees'] = pd.concat([R_dico['variables continues gardees'],keep_tb_q], axis = 0)
        R_dico['variables continues jetees'] = pd.concat([R_dico['variables continues jetees'],drop_tb_q], axis = 0)  
        R_dico['variables discretes gardees'] = pd.concat([R_dico['variables discretes gardees'],keep_tb_d], axis = 0)
        return E, R_dico
        
        
    elif method == 'Cramer':
        for col in E.columns:
            # Convert to string
            #♀corr.quantify_col(E[col], treat_na_as_zero = False)
                        
            index = E[col].apply(np.isreal)
            index = index & ~E[col].isnull()
            E[col][index] = E[col][index].astype(int)
            E[col] = E[col].astype(str)
            
            # Cramer
            index = ~Y.isnull()
            if len(E.loc[index,col].unique()) > 1:
                R, bool_R =  corr.get_correlation(E.loc[index,col], Y.loc[index], 1, 1)                
                if (col in dic)== True:
                    R=dic[col][1]
                    vprint('garde ' + col + ' avec R2 = ' + str(round(R,3)))
                    keep_tb.loc[keep_index] = [col,R]
                    keep_index = keep_index + 1
                    replace_dico = {}
                    for value in E[col].unique():
                        index = (E[col] == value)
                        replace_dico[value] = round(Y.loc[index].mean(),0)
                    G = pd.concat([G,E[col].replace(replace_dico)], axis =1)                                
                elif R < R_min:
                    vprint('vire ' + col + ' avec R2 = ' + str(round(R,3)))
                    drop_tb.loc[drop_index] = [col,R]
                    drop_index = drop_index + 1
                else:
                    vprint('garde ' + col + ' avec R2 = ' + str(round(R,3)))
                    keep_tb.loc[keep_index] = [col,R]
                    keep_index = keep_index + 1
                    replace_dico = {}
                    for value in E[col].unique():
                        index = (E[col] == value)
                        replace_dico[value] = round(Y.loc[index].mean(),0)
                    G = pd.concat([G,E[col].replace(replace_dico)], axis =1)
                    
            else:
                vprint('vire ' + col + ' car valeurs constantes')
                #tdc.drop(i, axis = 1, inplace = True)
                drop_tb.loc[drop_index] = [col,0]
                drop_index = drop_index + 1
        
        drop_tb.sort_values(by = 'R2', inplace = True, ascending = False)
        keep_tb.sort_values(by = 'R2', inplace = True, ascending = False)
        R_dico['variables continues gardees'] = pd.concat([R_dico['variables continues gardees'],keep_tb], axis = 0)
        R_dico['variables continues jetees'] = pd.concat([R_dico['variables continues jetees'],drop_tb], axis = 0)
        return G, R_dico     
    
    
    else:
        raise ValueError('methode non reconnue')

# =============================================================================
#                              Correlation
# =============================================================================


## Calcul des groupes de variables correlées entre elles
#  @brief Calcule la correlation entre les variables explicatives
#  @details Elle crée tableau des groupes de variables correllées entre elles
#  @param T: dataframe : tableau des variables explicatives
#  @param R_dico: dictionnaire: dictionnaire regroupant les R des variables explicatives
#  @param seuil_cramer: float : seuil à partir du quel des variables discretes sont considérées comme corrélées
#  @param seuil_corr: float : seuil à partir du quel des variables continues sont considérées comme corrélées
#  @return correlation_tb: dataframe : tableau des groupes de variables correllées
#  @date 29 septembre 2017
#  __________________________________________________________________________________________________


def correlation(T, R_dico, seuil_cramer, seuil_corr):
    corr_matrix, corr_matrix_bool = corr.correlation_matrix(T, seuil_cramer, seuil_corr)
    
    column_name = corr_matrix_bool.columns.values
    corel_arr = corr_matrix_bool.values
    
    dico = {}
    colonne_parcourue_arr = []
    for i in range(0, len(column_name)):
        colonne_parcourue_arr.append(False)
    dico = corr.fill_dico_by_corr(colonne_parcourue_arr, corel_arr, dico, column_name)
    
    correlation_tb = pd.DataFrame()
    for key in dico.keys():
        list_r = []
        for i in dico[key]:
            if len(R_dico['variables continues gardees'].loc[R_dico['variables continues gardees'] 
            ['col_name'] == i, 'R2'].values) == 1:
                list_r.append(R_dico['variables continues gardees'].loc[R_dico['variables continues gardees']
                ['col_name'] == i, 'R2'].values[0])
            elif len(R_dico['variables discretes gardees'].loc[R_dico['variables discretes gardees']
            ['col_name'] == i, 'R2'].values) == 1:
                list_r.append(R_dico['variables discretes gardees'].loc[R_dico['variables discretes gardees']
                ['col_name'] == i, 'R2'].values[0])
            else:
                raise ValueError('Cannot find R2 for: ' + i)
        a = pd.DataFrame({'groupe_variables_correlees': [dico[key]], 'variable_representante':
            dico[key][np.argmax(list_r)],'R2': list_r[np.argmax(list_r)]})
        correlation_tb = pd.concat([correlation_tb,a], axis = 0)
    
    return correlation_tb


# =============================================================================
#                               Normalise
# =============================================================================


## Normalisation des variables continues 
#  @brief Les variables continues sont normalisées et pondérées par leur coefficient de corellation
#  @details 
#  @param E: dataframe : dataframe des colonnes à normaliser
#  @param R_dico: dictionnaire: dictionnaire regroupant les R des variables explicatives
#  @return E: dataframe : dataframe normalisé
#  @date 29 septembre 2017
#  __________________________________________________________________________________________________

def normalize_(E, R_dico):
    for col in E.columns:
        if sum(R_dico['variables continues gardees']['col_name'] == col) == 1:
            index = (R_dico['variables continues gardees']['col_name'] == col)
            R = R_dico['variables continues gardees'].loc[index,'R2'].values[0]
            
        elif sum(R_dico['variables discretes gardees']['col_name'] == col) == 1:
            index = (R_dico['variables discretes gardees']['col_name'] == col)
            R = R_dico['variables discretes gardees'].loc[index,'R2'].values[0]
        elif sum(R_dico['variables']['col_name'] == col) == 1:
            index = (R_dico['variables']['col_name'] == col)
            R = R_dico['variables'].loc[index,'R2'].values[0]
        else:
            raise ValueError('ne trouve pas le R2 correspondant à la colonne')
        
        E[col] = (E[col] - E[col].mean()) / E[col].std() * R
    return E

# =============================================================================
#                                   Main
# =============================================================================

## Fonction main de la datapréparation
#  @brief Supprime tous les fichiers dans result & Charge les données, crée de nouvelles variables, séléctionne les variables et les normalise.
#  @details 
#  @param fichier: fichier csv contenant les variables explicatives
#  @param R_dico: dictionnaire regroupant les R des variables explicatives
#  @param fichier_sortie: fichier csv contenant la variable expliquee
#  @param taille_ech: nombre de lignes du fichier de variables explicatives
#  @param R_cont_y: seuil à partir du quel nous considérons qu'une variable explicative continue est corrélée avec la sortie
#  @param R_Cramer_y: seuil à partir du quel nous considérons qu'une variable explicative discrete est corrélée avec la sortie
#  @param R_cont_x: seuil à partir du quel nous considérons que deux variables explicatives continues sont correllées
#  @param R_Cramer_x: seuil à partir du quel nous considérons que deux variables explicatives discretes sont correllées
#  @param verbose: afficher l'avancee de la fonction ou non
#  @return T: dataframe des variables explicatives transformées
#  @return Y: sortie
#  @return R_dico: dictionnaire regroupant les R des variables explicatives
#  @date 17 Octobre 2017
#  __________________________________________________________________________________________________

def main(fichier = 'variables_explicatives.csv',fichier_sortie = 'zz.csv', method_disc='regression',method_continuous='regression',
            taille_ech = 50000,R_cont_y = 0.3, R_Cramer_y = 0.25, R_cont_x = 0.8, R_Cramer_x = 0.7,dic={},
            normalize = True, verbose = True, path_rslt='chemin_vers_donnees', suffix_table='suffix_table'):

    ######################################
    #Delete all file in the folder result#
    ######################################
    print('Do not delete all file in the folder result! No need for the moment! ')
    #path_del=path_rslt + "*"
    #r = glob.glob(path_del)
    #for i in r:
    #   os.remove(i)
    
    ############################
    # Définition des paramètres
    ############################
    print('Charge données.................')
    T = charge_expl(fichier = fichier, taille_ech = taille_ech,
                    drop = [], index_col = '﻿IDCLI_CALCULE') #﻿
    Y = charge_sortie(T = T, fichier_sortie = fichier_sortie,
                      input_col = 'revenu', index_col = '﻿IDCLI_CALCULE')
    
    #####################################
    #Détecte variables quali et quanti ## 
    #####################################    
    discrete_column, quantitative_column = get_column_type(T, seuil = 12)
    
    R_dico = { 'variables discretes jetees': None,
               'variables continues jetees': None,
               'variables discretes gardees': None,
               'variables continues gardees': None,
               'variables': None                
               }

    ###################################################
    #Préparation et selection des variables discretes #
    ###################################################
    print('Preparation et selection des variables discretes...')
    F, R_dico = treat_discrete_columns(T[discrete_column], Y,
                R_dico,dic, method = method_disc, R_min = R_Cramer_y, verbose = verbose)
    T.drop(discrete_column, axis = 1, inplace = True)
    
    #######################################################
    #Préparation et selection des variables quantitatives #
    #######################################################
    print('Preparation et selection des variables continues...')
    E, R_dico = treat_continuous_columns(T[quantitative_column], Y,
                R_dico,dic, method = method_continuous, R_min = R_Cramer_y, R_cont_y = R_cont_y,
                R_Cramer_y = R_Cramer_y, verbose = verbose)
    T.drop(quantitative_column, axis = 1, inplace = True)
    
    #################################
    # Merge variables and R2 tables #
    #################################
    T = pd.concat([E,F], axis = 1)
    #del E, F
    
    T.drop([col for col in R_dico['variables']['col_name'] if col in T], axis=1, inplace=True)
    ###############
    # Correlation #
    ###############
    print('Calcul des correlations...')
    correlation_tb = correlation(T, R_dico, seuil_cramer = R_Cramer_x, seuil_corr = R_cont_x)
    R_dico['groupe variables'] = correlation_tb
    A = T[correlation_tb['variable_representante']]

    T = pd.concat([E,F], axis = 1)
    del E, F
    
    for col in R_dico['variables']['col_name']:
        if (col in A.columns)==False:
            A=pd.concat([A,T[col]],axis=1)
    
    
#    for col in dic.keys():
#        if (col in T.columns)==True and (col in A.columns)==False:
#            A=pd.concat([A,T[col]],axis=1)
#        if ('log_' + col in T.columns)==True and ('log_' + col in A.columns)==False:
#            A=pd.concat([A,T['log_' + col]],axis=1)
#        if ('nan_' + col in T.columns)==True and ('nan_' + col in A.columns)==False:
#            A=pd.concat([A,T['nan_' + col]],axis=1)
    ########################
    # Normalize and fill na#
    ########################
    if normalize == True:
        A = normalize_(A,R_dico)
        A.fillna(0, inplace = True)
    
    return A, Y, R_dico

