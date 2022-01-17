import argparse
import pandas as pd


def prepare_data_files(arguments: argparse.Namespace):
    """
    Function to prepare and save all required data files:
        - genotype matrix in unified format as .h5 file with,
        - phenotype matrix in unified format as .csv file,
        - file containing maf filter and data split indices as .h5.
    :param arguments: all arguments specified by the user
    """
    # genotype matrix, die vom User angegeben wurde, laden und prüfen
    ... = check_transform_format_genotype_matrix(arguments=arguments) #Rückgabewerte solltest du dir überlegen, die du für den weiteren Ablauf brauchst
    ... = check_transform_format_phenotype_matrix(arguments=arguments)
    ... = genotype_phenotype_matching(arguments=arguments, genotype_matrix=genotype_matrix,
                                      phenotype_matrix=phenotype_matrix) # ANNAHME gen und pheno matrix werden von den obigen zurückgegeben
    check_create_index_file(arguments=arguments) #, sample_ids und was du alles so brauchst)


def check_transform_format_genotype_matrix(arguments: argparse.Namespace):
    """
    Function to check the format of the specified genotype matrix.
    Unified genotype matrix will be saved in subdirectory data and named NAME_OF_GENOTYPE_MATRIX.h5
    Unified format of the .h5 file of the genotype matrix required for the further processes:
    # TODO MAURA: Format beschreiben
    :param arguments: all arguments specified by the user
    :return: # TODO: angaben
    """
    # prüfen ob Datei schon im .h5 vorliegt
    ## wenn .h5, dann Aufbau der .h5 überprüfen #TODO MAURA: wenn der Nutzer schon eine .h5 übergibt, aber das Format nicht zu unserem passt, haben wir ein Problem mit dem speichern, weil dann biede Dateien gleich heißen --> sollen wir bei uns noch irgendeine suffix/prefix einbauen, analog bei der phenotype matrix?
    ## Falls Aufbau nicht passt oder andrees Format --> in unser Format übertragen und als .h5 speichern


def check_transform_format_phenotype_matrix(arguments: argparse.Namespace):
    """
    Function to check the format of the specified phenotype matrix.
    Unified genotype matrix will be saved in subdirectory data and named NAME_OF_PHENOTYPE_MATRIX.csv
    Unified format of the .csv file of the phenotype matrix required for the further processes:
    # TODO MAURA: Format beschreiben
    :param arguments: all arguments specified by the user
    :return: # TODO: angaben
    """
    # prüfen ob Datei schon als .csv vorliegt
    ## wenn .csv, dann Aufbau der .csv überprüfen #TODO MAURA: wenn der Nutzer schon eine .csv übergibt, aber das Format nicht zu unserem passt, haben wir ein Problem mit dem speichern, weil dann biede Dateien gleich heißen --> sollen wir bei uns noch irgendeine suffix/prefix einbauen, analog bei der phenotype matrix?
    ## Falls Aufbau nicht passt oder andrees Format --> in unser Format übertragen und als .csv speichern

def genotype_phenotype_matching(arguments: argparse.Namespace, genotype_matrix: pd.DataFrame,
                                phenotype_matrix: pd.DataFrame): #TODO: passen die Dateiformate?
    """
    Function to match the handed over genotype and phenotype matrix for the phenotype specified by the user.
    :param arguments: all arguments specified by the user
    :param genotype_matrix: genotype matrix in unified format
    :param phenotype_matrix: phenotype matrix in unified format
    :return: # TODO: angaben
    """
    # TODO: MAURA
    # Matchen und Werte für weiteren Teil vom Code zurückgeben


def check_create_index_file(arguments: argparse.Namespace): #TODO: weitere Übergabeparmaeter hinzufügen
    """
    Function to check the .h5 file containing the maf filters and data splits for the combination of genotype matrix,
    phenotype matrix and phenotype.
    It will be created using standard values for the maf filters and data splits in case it does not exist.
    Otherwise, the maf filter and data splits specified by the user are checked for existence.
    Unified format of .h5 file containing the maf filters and data splits:
        #TODO: FORMAT BESCHREIBEN - siehe base_dataset
    Standard values for the maf filters and data splits:
        #TODO: beschreiben
    :param arguments: all arguments specified by the user
    :return:
    """
    # Prüfen ob die Datei schon existiert
    ## Wenn ja, dann prüfen ob die MAF Filter und Data Splits schon existieren, die der User sich wünscht
    # --> würde vmtl. Sinn machen gleich Unterfunktionen anzulegen, die zu einer .h5 Maf Filter bzw. data splits mit bestimmten Übergabeparametern anlegen
    ## Wenn nein, dann erstellen mit Standard-Werten. NICHT VERGESSEN: Alle Werte, die für spätere MAF Filter und data splits relevant sind, mit abspeichern
    # TODO: die Funktion könnte recht groß werden, ggfs. macht es Sinn das noch weiter zu unterteilen
