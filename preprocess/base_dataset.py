import argparse


class Dataset:
    """Class containing dataset ready for optimization (e.g. geno/phenotype matched)"""

    def __init__(self, arguments: argparse.Namespace):
        self.test_set_indices = ... # nur ein array, sofern wir für jeden outer fold neu laden. ansonsten liste.
        self.train_set_indices = ... # LISTE (bei train val nur ein element, ansonsten 5) # bei nested verschachtelte Liste
        self.val_set_indices = ... # LISTE (bei train val nur ein element, ansonsten 5) # bei nested verschachtelte Liste
        self.X_012_encoded, self.y_012_encoded = self.match_maf_filter_raw_data(arguments=arguments, coding='012')
        self.X_XXX_encoded, self.y_XXX_encoded = self.match_maf_filter_raw_data(arguments=arguments, coding='XXX')

    def match_maf_filter_raw_data(self, arguments: argparse.Namespace, coding: str):
        # TODO: MAURA - Rohdaten in richtiger Codierung laden, matchen, maf filtern und X sowie y getrennt zurückgeben.
        #  Datentyp?
        return ...
