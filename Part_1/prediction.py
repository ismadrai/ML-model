
## @package prediction
#  Module de calcul des prédictions.
#  @author Isma
#  @version 1.1.0
#  @date 23 Octobre 2017

# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from math import *
import scipy as sc
from scipy.stats import t
from numpy import linalg
from sklearn.cross_validation import train_test_split
import numpy as np 


## Documentation de la fonction de prédiction depuis chaque cluster
#  @brief Fonction prediction_from_cluster
#  @details Prédiction de la variable à expliquer par differentes modèlisations
#  @param dfX : variables explicatives du modèle
#  @param Y : variable à expliquer
#  @param index_no_test : indices de l'echantillon de test
#  @param index_test : indices de l'echantillon de validation
#  @param method : méthode de prédiction appliquée
#  @return Y_pred : Le Revenu prédit
#  @date 24 Octobre 2017

#  Fonction de prediction depuis chaque cluster

def prediction_from_cluster(dfX, Y, index_no_test, index_test, method):
    if sum(~Y[index_no_test].isnull()) == 0:
        return float('Nan')
    if len(index_test) == 0:
        return float('Nan')
    else:            
        if method == 'mean':
            print("0****************mean")
            return int(Y.loc[index_no_test].mean())
        elif method == 'regression':
            print("0****************regression")
            lr = LinearRegression()
            index = Y[index_no_test][~Y[index_no_test].isnull()].index
            lr.fit(dfX.loc[index],Y[index])
            Y_pred = lr.predict(dfX.loc[index_test])
            return Y_pred
        elif method == 'random_forest':
            print("0****************random_forest")
            rf = RandomForestRegressor(n_estimators=10, criterion='mse')
            index = Y[index_no_test][~Y[index_no_test].isnull()].index
            rf.fit(dfX.loc[index],Y[index])
            Y_pred = rf.predict(dfX.loc[index_test])
            return Y_pred
        elif method == 'Ridge':
            print("0****************Ridgeregression")
            rd = Ridge (alpha = .5)
            index = Y[index_no_test][~Y[index_no_test].isnull()].index
            rd.fit(dfX.loc[index],Y[index])
            Y_pred = rd.predict(dfX.loc[index_test])
            return Y_pred  
        else:
            raise ValueError('unknown method')



## Documentation de la fonction de création de la table des résultats de la prédiction
#  @brief Fonction get_result
#  @details Création de la table des résultats de la prédiction
#  @param Y_real : Variable à expliquer réelle 
#  @param Y_pred : Variable à expliquer prédite
#  @param abs error : Valeur de l'erreur absolue
#  @param relative error : Valeur de l'erreur relative
#  @param cluster : Numero du cluster 
#  @param max_cluster : Valeur maximale de Y_pred du cluster
#  @param min_cluster : Valeur minimale de Y_pred du cluster
#  @param largeur : Largeur du cluster
#  @param number_non_na : Nombre de valeurs manquants (Variable à expliquer)
#  @return result : Table des résultats de la prédiction
#  @date 24 Octobre 2017
#
#  Fonction de création de la table des résultats de la prédiction


def get_result(repartition, dfX, Y, Y_test, method):
    result = pd.DataFrame(columns = ['Y_real','Y_pred','abs error','relative error','cluster',
                'max_cluster', 'min_cluster','largeur','number_non_na'],index = Y_test.index)
    for cluster in repartition.keys():
        df_cluster = repartition[cluster]
        index_test = df_cluster[df_cluster.index.isin(Y_test.index)].index
        index_no_test = df_cluster[~df_cluster.index.isin(Y_test.index)].index
        result.loc[index_test,'Y_real'] = df_cluster.loc[index_test,'revenu'].astype(int)
        #result.loc[index_test,'Y_pred'] = prediction_from_cluster(df_cluster.loc[index_no_test])
        result.loc[index_test,'Y_pred'] = prediction_from_cluster(dfX, Y,
                    index_no_test, index_test, method = method)
        result.loc[index_test,'abs error'] = abs(result.loc[index_test,'Y_real'] - \
                                            result.loc[index_test,'Y_pred'])
        result.loc[index_test,'relative error'] = 100 * abs(result.loc[index_test,'Y_real'] - \
                    result.loc[index_test,'Y_pred']) / result.loc[index_test,'Y_real']        
        result.loc[index_test,'cluster'] = cluster
        result.loc[index_test,'max_cluster'] = df_cluster.loc[index_no_test,'revenu'].max()
        result.loc[index_test,'min_cluster'] = df_cluster.loc[index_no_test,'revenu'].min()
        result.loc[index_test,'largeur'] = result.loc[index_test,'max_cluster'] - \
                                            result.loc[index_test,'min_cluster']
        result.loc[index_test,'number_non_na'] = sum(~df_cluster.loc[index_no_test,'revenu'].isnull())
    return result



## Documentation de la fonction de prédiction depuis chaque cluster
#  @brief Fonction predi_reg
#  @details Prédiction de la variable à expliquer par le modèle de regression linéaire
#  @param dfX : variables explicatives du modèle
#  @param Y : variable à expliquer
#  @param index_train : indices de l'echantillon de test
#  @param index_test : indices de l'echantillon de validation
#  @param df_cluster_index : numéro du cluster
#  @return Y_pred : Le Revenu prédit
#  @date 24 Octobre 2017
#
#  Fonction de prediction depuis chaque cluster


# def predi_reg(dfX, Y, index_train, index_test,df_cluster_index):
    # lr = LinearRegression()       
    # lr.fit(dfX.loc[index_train],Y[index_train])
    # Y_pred_test = lr.predict(dfX.loc[index_test])
    # Y_pred = lr.predict(dfX.loc[df_cluster_index])
    # return Y_pred_test,Y_pred 


def predi_reg(dfX, Y, index_train, index_test,df_cluster_index):
    rd = Ridge (alpha = .5)       
    rd.fit(dfX.loc[index_train],Y[index_train])
    Y_pred_test = rd.predict(dfX.loc[index_test])
    Y_pred = rd.predict(dfX.loc[df_cluster_index])
    return Y_pred_test,Y_pred 


## Documentation de la fonction de calcul de l'intervalle de confiance et sortie des résultats
#  @brief Fonction IC_reg
#  @details Calcul de l'intervalle de confiance et sortie des résultats
#  @param dfX : variables explicatives du modèle
#  @param Y : variable à expliquer
#  @param repartition : repartition des données par cluster
#  @param path_rslt : repertoire des résultats
#  @param suffix_table : suffix du fichier
#  @return result : Sortie du fichier avec les résultats finaux
#  @return result_test : Sortie du fichier avec les résultats pour Y_real connu
#  @date 24 Octobre 2017
#  
#  Fonction de calcul de l'intervalle de confiance et sortie des résultats

def IC_reg(repartition, dfX, Y,path_rslt, suffix_table):
    u = pd.DataFrame(data = 1, columns = ['constante'] , index = dfX.index)
    dfX = pd.concat([u,dfX], axis = 1)
    result = pd.DataFrame(columns = ['Y_real','Y_pred','error','cluster','min_IC','max_IC','largeur','% largeur'],index=Y.index)
    result_test = pd.DataFrame(columns = ['Y_real','Y_pred','error','cluster','min_IC','max_IC','largeur','% largeur'],index=Y.index)
    
    for j in repartition.keys():
        #try:
        df_cluster=repartition[j]
        #intialisation et repartition train et test
        #recuperation des indexes des revenus renseignés
        index = df_cluster["revenu"].index[~df_cluster["revenu"].apply(np.isnan)]
        dfX_train, dfX_test, Y_train, Y_test = train_test_split(dfX.loc[index], Y[index],test_size = 0.4, random_state = 44)
        index_test = Y_test.index
        index_train = Y_train.index
        #except:
        #    print("Le programme plante au cluster " + str(j))
        #condition sur les clusters
        if len(df_cluster)>= 40 and df_cluster["revenu"].isnull().sum()/len(df_cluster) <1:
            df_cluster_index=df_cluster.index             
            result_test.loc[index_test,'cluster']=j
            result.loc[df_cluster_index,'cluster']=j
            #calcul des fonctions de prévision et erreur
            result.loc[df_cluster.index,'Y_real'] = df_cluster.loc[df_cluster.index,'revenu']
            result_test.loc[index_test,'Y_real'] = df_cluster.loc[index_test,'revenu'].astype(int)
            result_test.loc[index_test,'Y_pred'],result.loc[df_cluster_index,'Y_pred'] = predi_reg(dfX, Y,
                        index_train, index_test, df_cluster_index)        
            result_test.loc[index_test,'error'] = result_test.loc[index_test,'Y_pred']-result_test.loc[index_test,'Y_real']        
            result.loc[df_cluster_index,'error'] = result.loc[df_cluster_index,'Y_pred']-result.loc[df_cluster_index,'Y_real']      
            result['lib_segment'] = suffix_table
            
            n=len(df_cluster.loc[df_cluster_index]) #apprentisage ou test !!!!
            #np.linalg.matrix_rank(dfX.loc[index_train])
            df= n -(len(dfX.columns)-1) - 1
            quantile=t.interval(0.85, df)[1]        
            MSE = result_test.loc[index_test,'Y_pred'].std() #MSE de result_test ou de result !!!!
            X=np.dot(dfX.loc[index_test].T,dfX.loc[index_test])            
            if linalg.det(X) != 0:
                X=linalg.inv(X)
                for i in range(0,len(result_test.loc[index_test,'Y_pred'])):
                    a=np.matrix(dfX.loc[index_test][i:i+1])
                    u=(a*X)*(a.T)
                    #h=result.loc[index_test,'Y_pred'][i:i+1].index
                    if (1+u) > 0:
                        result_test.loc[result_test.loc[index_test,'Y_pred'][i:i+1].index,'min_IC']=result_test.loc[index_test,'Y_pred'][i:i+1] - (quantile * MSE * sqrt(1+u))
                        result_test.loc[result_test.loc[index_test,'Y_pred'][i:i+1].index,'max_IC']=result_test.loc[index_test,'Y_pred'][i:i+1] + (quantile * MSE * sqrt(1+u))
                    else:
                        print('cluster ' + str(j)+ ' contient valeur negative')
                        result_test.loc[result_test.loc[index_test,'Y_pred'][i:i+1].index,'min_IC'] = 0
                        result_test.loc[result_test.loc[index_test,'Y_pred'][i:i+1].index,'max_IC'] = 0
                result_test.loc[index_test,'largeur'] = result_test.loc[index_test,'max_IC'].astype(int)-result_test.loc[index_test,'min_IC'].astype(int)
                result_test.loc[index_test,'% largeur'] = result_test.loc[index_test,'largeur'].astype(int)/result_test.loc[index_test,'Y_pred'].astype(int)
            else:
                result_test.loc[index_test,'min_IC'] = 'Inv'
                result_test.loc[index_test,'max_IC'] = 'Inv'
                
            n=len(df_cluster.loc[df_cluster_index]) #apprentisage ou test !!!!
            #np.linalg.matrix_rank(dfX.loc[df_cluster_index])
            df= n -(len(dfX.columns)-1)- 1
            quantile=t.interval(0.85, df)[1]        
            MSE = result.loc[df_cluster_index,'Y_pred'].std() #MSE de result_test ou de result !!!!           
            X = np.matrix(dfX.loc[df_cluster_index]).T * np.matrix(dfX.loc[df_cluster_index])
            if linalg.det(X) != 0:
                X=linalg.inv(X)
                for i in range(0,len(result.loc[df_cluster_index,'Y_pred'])):
                    a=np.matrix(dfX.loc[df_cluster_index][i:i+1])
                    u=(a*X)*(a.T)
                    #h=result.loc[index_test,'Y_pred'][i:i+1].index
                    if (1+u) > 0:
                        result.loc[result.loc[df_cluster_index,'Y_pred'][i:i+1].index,'min_IC']=(result.loc[df_cluster_index,'Y_pred'][i:i+1] - (quantile * MSE * sqrt(1+u))).astype(int)
                        result.loc[result.loc[df_cluster_index,'Y_pred'][i:i+1].index,'max_IC']=(result.loc[df_cluster_index,'Y_pred'][i:i+1] + (quantile * MSE * sqrt(1+u))).astype(int)
                    else:
                        print('cluster ' + str(j)+ ' contient valeur negative')
                        result.loc[result.loc[df_cluster_index,'Y_pred'][i:i+1].index,'min_IC'] = 0
                        result.loc[result.loc[df_cluster_index,'Y_pred'][i:i+1].index,'max_IC'] = 0
                result.loc[df_cluster_index,'largeur'] = result.loc[df_cluster_index,'max_IC'].astype(int)-result.loc[df_cluster_index,'min_IC'].astype(int)  
                result.loc[df_cluster_index,'% largeur'] = result.loc[df_cluster_index,'largeur'].astype(int)/result.loc[df_cluster_index,'Y_pred'].astype(int)  
                
            else:
                result.loc[df_cluster_index,'min_IC'] = 'Inv'
                result.loc[df_cluster_index,'max_IC'] = 'Inv'
            
        else:
            print("cluster "+str(j)+" ne remplit pas les conditions")
    result_test = result_test[pd.notnull(result_test['cluster'])]
    result = result[pd.notnull(result['cluster'])]
    dfX.drop(['constante'], axis=1,inplace=True)
    #path="/mnt/smb/TAMPON/Igor/RFR/data_rslt/"
    #result_test.to_excel(path_rslt+"result_test" + suffix_table + ".xlsx",encoding="utf-8", index=True)
    #result.to_excel(path_rslt+"result"  + suffix_table + ".xlsx",encoding="utf-8", index=True)
    result.to_csv(path_rslt+"result"  + suffix_table + ".csv",sep=";",encoding="utf-8", index=True)
    return result,result_test

    

## Fonction de calcul de la matrice de confusion et sortie des résultats
#  @brief Création de la matrice de confusion
#  @details Fonction de calcul de la matrice de confusion et sortie des résultats
#  @param Y_real : array : Valeur réelle de la variable à expliquer
#  @param Y_pred : array : Valeur prédite de la variable à expliquer
#  @param threshold_pred : int :  Seuil de prédiction
#  @param threshold_rich : int :  Seuil réel
#  @return confusion_matrix_ : array : Restitution de la matrice de confusion
#  @date : 29 septembre 2017
#  
        
def get_score_classification(Y_real, Y_pred, threshold_pred, threshold_rich):    
    Y_real_cl = (Y_real > threshold_rich).astype(int)
    Y_pred_cl = (Y_pred > threshold_pred).astype(int)
    confusion_matrix_ = metrics.confusion_matrix(Y_real_cl, Y_pred_cl)
    recall = round(confusion_matrix_[1,1] / sum(confusion_matrix_[1,:]),4)
    precision = round(confusion_matrix_[1,1] / sum(confusion_matrix_[:,1]),4)
    print('recall: ' + str(recall))
    print('precision: ' + str(precision))
    return confusion_matrix_


## Fonction de restitution graphique des résultats
#  @brief Fonction classification_curve
#  @details Restitution graphique des résultats de la matrice de confusion, courbe ROC
#  @param Y_real : array : Valeur réelle de la variable à expliquer
#  @param Y_pred : array : Valeur prédite de la variable à expliquer
#  @param col : string : Choix de la couleur des courbes
#  @param threshold_rich : int :  Seuil réel
#  @return a,b : graph : Restitution graphique des résultats
#  @date : 29 septembre 2017
        
def classification_curve(Y_real, Y_pred, threshold_rich = 100000, col = 'white'):
    Y_test_b = (Y_real > threshold_rich).astype(int)
    Y_pred_p = Y_pred.astype(float)
    
    Y_test_b = Y_test_b[~Y_pred_p.isnull()]
    Y_pred_p = Y_pred_p[~Y_pred_p.isnull()]
    
    
    precision, recall, thresholds = metrics.precision_recall_curve(Y_test_b, Y_pred_p)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test_b, Y_pred_p)
    
    # fig = plt.figure(figsize = (15,5))
    # ax = fig.add_subplot(121)
    # ax.plot(recall, precision, label='model (area = %0.2f)' %metrics.auc(recall, precision))
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.spines['bottom'].set_color(col)
    # ax.spines['top'].set_color(col)
    # ax.xaxis.label.set_color(col)
    # ax.yaxis.label.set_color(col)
    # ax.tick_params(axis='y', colors=col)
    # ax.tick_params(axis='x', colors=col)
    # ax.set_xlabel('% of rich detected (recall)')
    # ax.set_ylabel('% of usefull contacts (precision)')
    # ax.set_title('Precision-Recall curve', color=col)
    # ax.legend(loc="best")
    # ax.grid(True)
    
    # ax = fig.add_subplot(122)
    # ax.plot(fpr, tpr, label='model (area = %0.2f)' %metrics.auc(fpr, tpr))
    # plt.plot([0, 1], [0, 1], 'k--', label = 'random classification')    
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.spines['bottom'].set_color(col)
    # ax.spines['top'].set_color(col)
    # ax.xaxis.label.set_color(col)
    # ax.yaxis.label.set_color(col)
    # ax.tick_params(axis='y', colors=col)
    # ax.tick_params(axis='x', colors=col)
    # ax.set_xlabel('false positive')
    # ax.set_ylabel('true positive')
    # ax.set_title('ROC curve', color=col)
    # ax.legend(loc="best")
    # ax.grid(True)
    # plt.show()
    b=metrics.auc(fpr, tpr)
    a=metrics.auc(recall, precision)
    return(a,b)
    


## Fonction de restitution graphique de l'influence des variables explicatives sur la prédiction du revenu
#  @brief Fonction de restitution graphique du revenu en fonction des variables explicatives
#  @details Revenu réel et prédit en fonction des variables explicatives
#  @param dfX : dataframe : Les variables explicatives
#  @param result : dataframe : Dataframe des prédictions de la variable à expliquer
#  @param col : string : couleur
#  @return  : graph : Restitution graphique des résultats
#  @date : 29 septembre 2017

# def graph_variable_influence(dfX, result, col):
    # fig, axes = plt.subplots(nrows=1, ncols=dfX.shape[1], figsize=(11,2))
    # for ax, column in zip(axes,dfX.columns):
        # if column == 'Y_pred':
            # continue
        # if column != dfX.columns[0]:
            # ax.set_yticklabels([])
        # else:
            # ax.set_ylabel('Revenu predit')
        # ax.tick_params(axis='y', colors=col)
        # ax.tick_params(axis='x', colors=col)
        # ax.spines['bottom'].set_color(col)
        # ax.spines['top'].set_color(col)
        # ax.xaxis.label.set_color(col)
        # ax.yaxis.label.set_color(col)
        # ax.set_xticklabels([])
        # ax.set_ylim([0,300000])
        # ax.set_xlabel(column, rotation='vertical')
        # ax.scatter(dfX.loc[result['Y_real'].index,column],result['Y_pred'],color='blue',s=5,edgecolor='none')
    # plt.suptitle('Revenu predit en fonction des variables explicatives', color = col)
    # plt.show()
    
    # fig, axes = plt.subplots(nrows=1, ncols=dfX.shape[1], figsize=(11,2))
    # for ax, column in zip(axes,dfX.columns):
        # if column == 'Y_pred':
            # continue
        # if column != dfX.columns[0]:
            # ax.set_yticklabels([])
        # else:
            # ax.set_ylabel('Revenu predit')
        # ax.tick_params(axis='y', colors=col)
        # ax.tick_params(axis='x', colors=col)
        # ax.spines['bottom'].set_color(col)
        # ax.spines['top'].set_color(col)
        # ax.xaxis.label.set_color(col)
        # ax.yaxis.label.set_color(col)
        # ax.set_xticklabels([])
        # ax.set_ylim([0,300000])
        # ax.set_xlabel(column, rotation='vertical')
        # ax.scatter(dfX.loc[result['Y_real'].index,column],result['Y_real'],color='red',s=5,edgecolor='none')
    # plt.suptitle('Revenu reel en fonction des variables explicatives', color = col)
    # plt.show()


## Fonction de restitution de l'erreur absolue et relative de la modélisation
#  @brief Fonction get_regression_score
#  @param Y_real : array : Le revenu réel
#  @param Y_pred : array : Le revenu prédit
#  @return sortie : tuple : La fonction renvoie 2 éléments:
#                  - error_abs : float :  Erreur absolue   
#                  - error_rel : float : Erreur relative
#  @date : 29 septembre 2017

def get_regression_score(Y_real, Y_pred):
    # Delete outliers
    Y_real = Y_real[Y_real > 300]
    Y_pred = Y_pred[Y_real.index]
    error_abs = abs(Y_real - Y_pred)
    error_rel = ((abs(Y_real - Y_pred) / Y_real) * 100).astype(float).round(1)
    return int(error_abs.mean()), int(error_rel.mean())


## Fonction de restitution graphique de l'estimation du revenu vs revenu reel
#  @brief Fonction de restitution graphique de l'estimation du revenu vs revenu reel
#  @param Y_real : array : Le revenu réel
#  @param Y_pred : array : Le revenu prédit
#  @param col : string : couleur
#  @return sortie : graph : Restitution graphique des résultats
#  @date : 29 septembre 2017

# def regression_graph(Y_real, Y_pred, col):
    # fig = plt.figure(figsize = (15,5))
    # ax = fig.add_subplot(121)
    # ax.spines['bottom'].set_color(col)
    # ax.spines['top'].set_color(col)
    # ax.xaxis.label.set_color(col)
    # ax.yaxis.label.set_color(col)
    # ax.tick_params(axis='y', colors=col)
    # ax.set_ylim([0,300000])
    # ax.set_title('Estimation du revenu vs revenu reel', color = col)
    # x = range(0,len(Y_real))
    # index = Y_real.sort_values(inplace = False).index
    # ax.set_ylabel('Revenu')
    # ax.set_xlabel('Client')
    # #ax.scatter(x, Y_pred[index], color = 'r', label = 'Revenu estime')
    # ax.plot(x, Y_pred[index], color = 'r', alpha = 0.1)
    # ax.plot(x, Y_real[index],  label = 'Vrai revenu')
    # ax.grid()
    # ax.legend(loc = "best")
    
    # ax2 = fig.add_subplot(122)
    # ax2.scatter(Y_real, Y_pred, alpha = 0.1)
    # ax2.set_xlim((0,300000))
    # ax2.set_ylim((0,300000))
    # ax2.spines['bottom'].set_color(col)
    # ax2.spines['top'].set_color(col)
    # ax2.xaxis.label.set_color(col)
    # ax2.yaxis.label.set_color(col)
    # ax2.tick_params(axis='y', colors=col)
    # ax2.tick_params(axis='x', colors=col)
    # ax2.set_xlabel('revenu reel')
    # ax2.set_ylabel('revenu predit')
    # ax2.set_title('revenu prediction vs reel', color=col)
    # ax2.legend(loc="best")
    # ax2.grid(True)
    
    # plt.show()

