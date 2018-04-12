## @package K_means
#  Module de clustering par k-means.
#  Consiste à former des cluster d'individus et de s'assurer que chaque cluster possède assez de VAE connues.
#  @version 1.1.0
#  @date 17 Octobre 2017
#  @author Isma

## Documentation de la méthode de K-means class

# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import KMeans
import math
import numpy as np
import pdb


## Fonction de calcul des distances euclidiennes
#  @brief Implémentation du calcul des distances au centre des clusters 
#  @details La fonction calcule la distance euclidienne de chaque individus par rapports aux différents centroides qui ont été trouvés
#  @param tab_A : dataframe des coordonnées des individus
#  @param tab_B :dataframe des coordonnées des centroides des clusters
#  @return dist : float de la racine carré de la distance euclidienne par rapport à un centroid


# implémentation distance euclidienne
def euclidean_distance(tab_A, tab_B):
    dist = 0
    for i in range(len(tab_A)):
        dist += (tab_A[i] - tab_B[i]) ** 2
    return math.sqrt(dist)


## Clustering par k-means
#  @brief Construction des clusters par k-means
#  @details La fonction construit les clusters en deux étapes. Dans une prmière partie elle réalise un regroupement simple des individus par k-means; ensuite elle s'assure que chaque cluster formé précédemment possède assez de VAE connues
#  @param dfX : dataframe : dataframe des variables explicatives sur lesquelles on réalise la classification 
#  @param Y: dataframe:  dataframe de la variable à expliquer (VAE)
#  @param nb_clusters_init : integer : nombre de clusters à créer
#  @param methode_prediction : string : indique le nombre minimum d'individus dans chaque cluster. Si methode_prediction="mean" alors chaque cluster doit avoir au moins 10 individus
#  @return repartition : dictionnaire : dictionnaire contenant la répartition des k clusters
#  @date  juin 2017

# implémentation distance euclidienne
# Fonction de calcul des distances euclidiennes
def K_means(dfX,Y,nb_clusters_init, methode_prediction):
    """
    Realise K-means
    Input:
        - dfX : variables explicatives sur lesquelles en classifies  
        - Y: variable à expliquer 
        - nb_clusters : nombre de clusters
    Output:
        - repartition : dictionnaire contenant la répartition des k clusters 
    """
    # Using sklearn
    ## @internal
    ## Documentation for a method.
    #  @param self The object pointer.
    km = KMeans(n_clusters=nb_clusters_init, init="k-means++", random_state = 44) #, n_jobs = 4 
    #km = KMeans(n_clusters=k, init='k-means++',n_init=10, verbose=1) 
    km.fit(dfX)
    # Get cluster assignment labels

    labels = km.labels_
    centroids = km.cluster_centers_

###############################################################################
############################### RE-CLUSTERISATION #############################
###############################################################################
    #faire nbr de données connus / nb_cluster
    nb=(len(Y) - Y.isnull().sum())/nb_clusters_init
    
    if methode_prediction =='mean':
        nb_point_min = 10 #le nombr point connu
    else : 
        nb_point_min = int(nb)
        
    tempX = dfX.copy()
    tempX['cluster'] = labels
    tempX['IDCLI_CALCULE'] = tempX.index
    
    tempY = pd.DataFrame(Y)
    tempY['cluster'] = labels
        
    labels = pd.DataFrame(labels, index = dfX.index, columns = ['cluster'])
        
    reclust_liste = [] # liste des clusters à supprimer
    centro = pd.DataFrame(centroids)
    centro['num_center'] = centro.index # rangement du numéro de cluster dans les colonnes
                                        # pour y accéder quand ce sera un tableau
    
    for i in range(nb_clusters_init):      
        # Condition pour que le cluster soit re-réparti
        if sum(~tempY[tempY['cluster']==i]['sortie'].isnull()) < nb_point_min:
            temp = tempX[tempX.cluster==i]
            temp = np.array(temp)
            reclust_liste.append(temp)
            centro.drop([i], inplace = True, axis = 0)
            
    centro = np.array(centro) # passage en np.array pour accélerer
    
    for clust in reclust_liste:
        for i in range(clust.shape[0]):
            distances = [] # listes des distances aux centres des clusters
            for j in range(centro.shape[0]):
                distances.append(euclidean_distance(clust[i,:-2], centro[j,:-1]))
            min_dist = np.min(distances) # récupération du plus proche
            index_min = distances.index(min_dist)
            indice = centro[index_min,-1] # récupération du numéro         
            labels['cluster'].loc[int(clust[i,-1])] = indice # remplacement de l'indicateur
    
    # réindexation des numéros de cluster
    labels.sort_values(['cluster'], axis=0, inplace = True)
    i = 0
    for val in labels['cluster'].unique():
        labels[labels['cluster']==val]=i
        i+=1
        
###############################################################################
###############################################################################
    
    # Format results as a DataFrame
    results = pd.DataFrame(labels, index=dfX.index)
    results.columns = ['cluster']
        
    dfX = pd.concat([results, dfX], axis = 1)
    
    Y = pd.concat([results, Y], axis=1)
    Y.columns = ['cluster','revenu']
    # Distances aux centres de classes des observations
    #dist = km.transform(dfX)
    
    ##############################
    ## répartition des clusters ##
    ##############################
    
    #répartition des clusters
    repartition={}
    for i in range(len(Y['cluster'].unique())):
        repartition[i]=Y[Y.cluster == i]   
    
    return(repartition)
    ## @endinternal
 
