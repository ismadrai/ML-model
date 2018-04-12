
## @package launch_distributed
#  Module de lancement du programme en distribué, sur les serveurs Cloud (via dask & map).
#  @details Etapes du traitement 
#       - Paramétrage 
#       - Création du cluster cloud (Scheduler & Workers)
#       - Connexion du Client au Scheduler 
#       - Création du RDD (Resilient Distributed Data)
#       - Execution du programme en distribué sur serveurs Cloud (via client.map)
#
#  @version 1.1.0
#  @date 29 Novembre 2017
#  @author Igor

#%matplotlib inline
import pandas as pd
import os
from openpyxl import load_workbook
from datetime import datetime
from sklearn.cross_validation import train_test_split
import numpy as np
import csv
import glob
import gc
from dask.distributed import Client                               #DASK


#******************************************************************************************
#                                     scheduler
#******************************************************************************************
#client=Client('10.70.133.27:8786')
client=Client('10.70.133.29:8786')


# scheduler_external  = False                               #DASK
# if scheduler_external:                                   #DASK
    # s_adress = input('Entrez adresse du Scheduler: ')
    # client = Client(str(s_adress))                       #DASK
    # print(client.scheduler_info)                         #DASK
# else:
    # client= Client()                                     #DASK
    # print(client.scheduler_info)                         #DASK

    
########################
##Debut de parametrage##
########################
## Documentation du main 
#  @brief Paramétrage

# Mois de modélisation
idmois="_201708"
segmmm=""

# Chemin vers le dossier ou sont les codes du projet
path="/mnt/smb/TAMPON/Igor/RFR_V1/"
path_code = path + "script/"

#path_data = "/mnt/smb/TAMPON/Partages/Moteur_industr/03_PROD/Restitution/EAS/"
path_data = path + "data_src/"
path_rslt = path + "data_rslt/"

os.chdir(path_code)

#import local packages on workers 
client.upload_file('K_means.py')
client.upload_file('IntConf.py')
client.upload_file('prediction.py')
client.upload_file('correlation.py')
client.upload_file('data_preparation.py')
client.upload_file('main2.py')

import main2 as workflow
import data_preparation
import K_means as K_means
import IntConf
import prediction


fich_deb = "ech_variables_explicatives" + idmois + ".csv" 
#fich_vae = "revenu_client" + segmmm + idmois + ".csv"

segm_file= path_data + fich_deb

#################################################
##Pour le  parametrage statistique voir le main2 
#################################################

########################
##Fin de parametrage##
########################


##########################################################
###### Lancement du traitement par paquet de données #####
##########################################################


begin_global = datetime.now()

#CREATION LISTE DE SEGMENTS
liste_segments= pd.read_csv(segm_file, sep=';', usecols=['ID_SGMT_RFR'], encoding='utf-8').ID_SGMT_RFR.unique()
liste_segments.sort()

#liste_segments= pd.read_csv(segm_file, sep=';', usecols=['SGMT_RFR'], encoding='utf-8')
#liste_segments= liste_segments.loc[(liste_segments['SGMT_RFR'] == 'BQ_SD') | (liste_segments['SGMT_RFR'] == 'INA')].SGMT_RFR.unique()

############################################
#######Fonction à distribuer sur le serveurs
############################################

## Fonction d'execution du flux des traitements par paquet
#  @brief Fonction d'execution du flux des traitements
#  @details Execution de programme dès l'importation des données et jusqu'à la modelisation et restitution des résultats
#  @param list_segm : liste des données à distribuer (RDD)
#  @author Igor
#  @date 27 Novembre 2017

def launch_distributed(list_segm) : 
    import main2 as workflow
    begin_distributed = datetime.now()
    X = pd.read_csv(segm_file, sep=';', encoding='utf-8')  #LOAD FILE
    fichier= X[X['ID_SGMT_RFR']==list_segm].drop(['revenu','X','Y','IRIS_code'],axis=1) #FILTER FILE + DROP VAE (variables_explicatives)
    lib_seg=fichier.SGMT_RFR.unique()
    print("Debut execution du paquet " + str(lib_seg) )
    #fichier= X[X['ID_SGMT_RFR'].str.contains(list_segm)].drop(['revenu','X','Y','IRIS_code'],axis=1) #FILTER FILE + DROP VAE (variables_explicatives)
    fichier_sortie = X[X['ID_SGMT_RFR']==list_segm].loc[:,['﻿IDCLI_CALCULE','revenu']] #FILTER FILE + KEEP VAE (revenu_client) +attention au symbole caché dans idcli
    taille_ech= len(fichier)
    segm_fid= "_" + "".join(lib_seg)
    suffix_table = "" + segm_fid + idmois
    fich_vae = "revenu_client" + suffix_table + ".csv"
    fich_deb = "variables_explicatives" + suffix_table + ".csv" 
    del X
    workflow.workflow(fichier=fichier,fichier_sortie=fichier_sortie, path=path, path_rslt=path_rslt, taille_ech=taille_ech, suffix_table=suffix_table, begin_global= begin_global, fich_vae=fich_vae,fich_deb=fich_deb )
    print("Fin execution du paquet " + str(lib_seg) )


client.map(launch_distributed , liste_segments)

delta_time = round((datetime.now() - begin_global).total_seconds(),1)
print('Temps de calcul total est de ' + str(delta_time) + 's')

