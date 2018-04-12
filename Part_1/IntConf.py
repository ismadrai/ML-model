## @package IntConf
#  Module de calcul des intervalles de confiance .
#  @details Réalise l'intervalle de confiance de prédiction et retourne un fichier contenant des statistiques descriptives sur chaque cluster
#  @author Ilyas
#  @version 1.1.0
#  @date 29 septembre 2017

import pandas as pd
import scipy.stats as stats
import numpy as np
from math import *


# def debug(fct):
    # def debugd(*args, **kw):
        # print('##############################')
        # print('Fonction : ', fct.__name__)
        # print('##############################')
        # print()
        # result = fct(*args,**kw)
        # print()
        # print('##############################')
        # return result
    # return debugd


## Fonction d'intervalle de confiance 
#  @brief Realise l'intervalle de confiance sur chaque cluster pour la variable à expliquer
#  @details Permet de disposer d'un intervalle de confiance précis cluster par cluster 
#  @param  repartition : dict : Dictionnaire contenant la répartition des k clusters   
#  @param  alpha : float : Degré de confiance demandé pour l'intervalle
#  @param  path : chemin du répertoire dans lequel sera sauvegardé les graphes
#  @param  version : version sur laquelle l'intervalle de cofiance est appliqué
#  @return P : dataframe : DataFrame contenant des states sur chaque cluster (Taille, nb NaN, Moyenne, écart-type, IC)  
#  @date 29 septembre 2017

def IntConf(repartition,alpha,path_rslt,suffix_table):

    ## Fonction d'écriture des résultats  
        #  @brief Ecrit dans un fichier les statistiques des clusters en les triant par le min de l'IC
        #  @details 
        #  @param repartition : dictionnaire : dictionnaire contenant la répartition des k clusters
        #  @param  disc : dict : dictionnaire contenant les statistiques des clusters     
        #  @param  path : string : chemin d'enregistrement du fichier
        #  @return sorties : tuple : Renvoie deux fichiers : 
        #                   - l'un contient les statistiques de chaque cluster 
        #                   - l'autre les clusters avec leurs IDCLI_CALCULE
    #@debug
    def sort_file(disc,path_rslt,n):
        disc_sort=sorted(disc, reverse=True)
        fichier = open(path_rslt+"InterConf"+ suffix_table + ".txt", "w")
        somme=0
        for m in disc_sort:
            i,c,l,mean,sigma,R,pourcen=disc[m][0],disc[m][1],disc[m][2],disc[m][3],disc[m][4],disc[m][5],disc[m][6]
            if (c-l)>10:
                fichier.write('\n \n Cluster : '+str(i)+'\n Taille : '+str(c)+'\n Nb NaN : '+str(l)+'\n pourcentage des NaN dans cluster :'+str(pourcen)+'\n Moyenne : '+str(mean)+'\n Std : '+str(sigma)+'\n IC : '+str(R)+'\n')
            # if c>n/len(disc_sort):
                # pd.DataFrame(repartition[i]).to_csv(path_rslt+"cluster"+str(i)+ str(suffix_table) +".csv",sep=";",encoding="utf-8", index=True)
            somme=somme+pourcen
        fichier.write('\n\n pourcentage total des NaN : '+str(somme/len(disc_sort)))
        fichier.close()

    teams = ["Cluster","Taille", "Nb NaN","% NaN dans cluster", "Moyenne","Std","min IC","max IC","Largeur"]
    liste={}
    disc={}
    n=0
    D=pd.Series()
    for i in repartition.keys():
        S = pd.DataFrame(repartition[i])
        if len(S) != 0 :
            n=n+len(S)
            for j in S.ix[:,1:S.shape[1]].columns:
                mean, sigma = np.mean(S[j]), np.std(S[j])
                m1=mean - 1.96*sigma*(1+1/np.sqrt(len(S)))
                m2=mean + 1.96*sigma*(1+1/np.sqrt(len(S)))
                #R = [str(round(m1,3)),str(round(m2,3))]
                R = str([(round(m1,3)),(round(m2,3))])
                l = len(S[j][S[j].isnull()==True])
            pourcen = l/S.shape[0]
            disc[m1]=[i,S.shape[0],l,mean,sigma,R,pourcen] 
            liste[i] = i,S.shape[0],l,pourcen,str(round(mean,2)),str(round(sigma,2)),str(round(m1,2)),str(round(m2,2)),str(round(m2-m1,3))
            D=pd.concat([D,pd.Series(liste[i])], axis = 1)
        #else :
        #    del repartition[i]
    P=D.transpose()
    P.columns = teams

	#####################
    #concatener
    R=pd.DataFrame(columns=['moyenne VAE','variance','min IC','max IC','Taille cluster','nb VAE connue'])
    for i in range(0,len(repartition)):
        repartition[i]['IDCLI_CALCULE'] = repartition[i].index
        R=pd.concat([R,repartition[i]],axis=0)
        #calcul
        index=R['cluster']==i
        taille = len(R[index])
        if taille != 0 :
            mean_VAE=R['revenu'][index].mean()
            std_VAE=R['revenu'][index].std()
            m1=mean_VAE - 1.96*std_VAE*(1+1/sqrt(taille))
            m2=mean_VAE + 1.96*std_VAE*(1+1/sqrt(taille))
            R['moyenne VAE'].loc[index]=mean_VAE
            R['variance'].loc[index]=std_VAE
            R['min IC'].loc[index]=m1
            R['max IC'].loc[index]=m2
            R['Taille cluster'].loc[index]=taille
            R['nb VAE connue'].loc[index]=len(R[index & ~R['revenu'].isnull()])
    ##R = R[['cluster','revenu','IDCLI_CALCULE','moyenne VAE','variance','min IC','max IC','Taille cluster','nb VAE connue']]
    ##R.to_csv(path_rslt+"Statistique_VAE" + str(suffix_table) + ".csv",sep=";",encoding="utf-8", index=False)
    P.to_excel(path_rslt+"Statistique" + str(suffix_table) + ".xlsx",encoding="utf-8", index=False)
    sort_file(disc,path_rslt,n)
    return(P)
 
