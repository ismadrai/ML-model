
## @package launch_spark
#
#  @version 1.1.0
#  @date 5 décembre 2017
#  @author Claire

import pandas as pd
#import os
#from openpyxl import load_workbook
#from datetime import datetime
#from sklearn.cross_validation import train_test_split
#import numpy as np
#import csv
#import glob
import gc


    
########################
##Debut de parametrage##
########################
## Documentation du main 
#  @brief Paramétrage

# Mois de modélisation
idmois="201710"
#segmmm=""

# Chemin vers le dossier ou sont les codes du projet
path="/mnt/smb/TAMPON/Partages/Projets/RFR/RFR_V1_2_0/Codes/"
path_code = path + "script/"
path_data = "/user/eta4491/"
#path_rslt = path + "data_rslt/"

os.chdir(path_code)



#fich_vae = "revenu_client_" + str(num) + "_" + str(idmois) + ".csv"


##########################################################
###### Lancement du traitement par paquet de données #####
##########################################################


#begin_global = datetime.now()


############################################
#######Fonction à distribuer sur le serveurs
############################################

## Fonction d'execution du flux des traitements par paquet
#  @brief Fonction d'execution du flux des traitements
#  @details Execution de programme dès l'importation des données et jusqu'à la modelisation et restitution des résultats
#  @param list_segm : liste des données à distribuer (RDD)
#  @author Igor
#  @date 27 Novembre 2017

def launch_spark(segm) : 
    fichier = path_data + "variables_explicatives_"+ str(segm) + "_" + str(idmois) + ".csv"
    lines=sc.textFile(fichier)
    parts=lines.map(lambda l:l.split(";"))
    df_spark=parts.toDF() # ajouter les noms de colonnes
    df = df_spark.toPandas()  
    #X = pd.read_csv("/mnt/BIP10/data_rslt/variables_explicatives_"+str(segm)+"_201710.csv",sep=";",encoding='utf-8',low_memory=False)
    print(df.head())
    (a,b)=df.shape
    print("Fin execution du paquet " + str(segm) + " : " + str(df.SGMT_RFR.unique()) )
    return(a,b,df)

rdd=sc.parallelize([1,2,3,4,5,6,7])


def lect_fic(num) : 
    fichier = path_data + "variables_explicatives_"+ str(num) + "_" + str(idmois) + ".csv"
    lines=sc.textFile(fichier)
    parts=lines.map(lambda l:l.split(";"))
    df_spark=parts.toDF() # ajouter les noms de colonnes
    # comptage des lignes
    #
    return(df_spark)

#(a,b,df)=lect_fic("/user/eta4491/variables_explicatives_ex.csv")
