
## @package launch
#  Module de lancement du programme en local par paquet (via boucle FOR).
#  @details Etapes du traitement 
#       - Paramétrage 
#       - Création des paquets de données 
#       - Execution du programme, paquet par paquet (via boucle FOR)
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


########################
##Debut de parametrage##
########################
## Documentation du main 
#  @brief Paramétrage

# Mois de modélisation
idmois="_201708"
#segmmm=""

# Chemin vers le dossier ou sont les codes du projet
path="/home/eta4493/Projets/RFR_V1/"
path_code = path + "script/"

#path_data = "/mnt/smb/TAMPON/Partages/RFR/matrice de donnees/"
path_data = path + "data_src/"
path_rslt = path + "data_rslt/"

# Import modules locaux

os.chdir(path_code)
import main2 as workflow
import data_preparation
import K_means as K_means
import IntConf
import prediction



fich_deb = "variables_explicatives" + idmois + ".csv" 
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
liste_segments= pd.read_csv(segm_file, sep=';', usecols=['SGMT_RFR'], encoding='utf-8').SGMT_RFR.unique()


for list_segm in liste_segments : 
    X = pd.read_csv(segm_file, sep=';', encoding='utf-8')  #LOAD FILE
    begin_distributed = datetime.now()
    print("Debut execution du paquet " + str(list_segm) )
    fichier= X[X['SGMT_RFR'].str.contains(list_segm)].drop(['revenu','X','Y','IRIS_code'],axis=1) #FILTER FILE + DROP VAE (variables_explicatives)
    fichier_sortie = X[X['SGMT_RFR'].str.contains(list_segm)].loc[:,['﻿IDCLI_CALCULE','revenu']] #FILTER FILE + KEEP VAE (revenu_client) +attention au symbole caché dans idcli
    taille_ech= len(fichier)
    segm_fid= "_" + list_segm
    suffix_table = segm_fid + idmois
    fich_vae = "revenu_client" + suffix_table + ".csv"
    fich_deb = "variables_explicatives" + suffix_table + ".csv" 
    del X
    workflow.workflow(fichier=fichier,fichier_sortie=fichier_sortie, path=path, path_rslt=path_rslt, taille_ech=taille_ech, suffix_table=suffix_table, begin_distributed= begin_distributed, fich_vae=fich_vae,fich_deb=fich_deb )
    print("Fin execution du paquet " + str(list_segm) )


delta_time = round((datetime.now() - begin_global).total_seconds(),1)
print('Temps de calcul total est de ' + str(delta_time) + 's')



