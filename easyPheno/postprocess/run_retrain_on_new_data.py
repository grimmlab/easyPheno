import argparse

from . import model_reuse
from ..preprocess import encoding_functions

if __name__ == "__main__":
    """
    Run to train a model on a new dataset using the hyperparameters that worked best for the specified model results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data_dir", type=str,
                        help="Provide the full path of the data directory that contains the geno- and phenotype "
                             "files you want to optimize on")
    parser.add_argument("-gm", "--genotype_matrix", type=str,
                        help="Provide the name of the genotype matrix you want to predict on")
    parser.add_argument("-pm", "--phenotype_matrix", type=str,
                        help="Provide the name of the phenotype matrix you want to predict on")
    parser.add_argument("-p", "--phenotype", type=str,
                        help="Provide the name of the phenotype you want to predict on")
    parser.add_argument("-sd", "--save_dir", type=str,
                        help="Define the save directory for the results.")
    parser.add_argument("-rd", "--results_directory_model", type=str,
                        help="Provide the full path of the directory where your results of the model "
                             "you want to use are stored")

    parser.add_argument("-enc", "--encoding", type=str, default=None,
                        help="specify the encoding to use. Caution: has to be a possible encoding for the model to use."
                             "Valid arguments are: " + str(encoding_functions.get_list_of_encodings()))
    # Preprocess Params #
    parser.add_argument("-maf", "--maf_percentage", type=int, default=0,
                        help="specify the minor allele frequency (as percentage value). "
                             "specify 0 if you do not want a maf filter.")
    parser.add_argument("-split", "--datasplit", type=str, default='nested-cv',
                        help="specify the data split to use: 'nested-cv' | 'cv-test' | 'train-val-test'"
                             "Default values are 5 folds, train-test-split to 80/20 and train-val-test to 60/20/20")
    parser.add_argument("-testperc", "--test_set_size_percentage", type=int, default=20,
                        help="specify the size of the test set in percentage. "
                             "Standard is 20, only relevant for 'cv-test' and 'train-val-test'")
    parser.add_argument("-valperc", "--val_set_size_percentage", type=int, default=20,
                        help="specify the size of the validation set in percentage. "
                             "Standard is 20, only relevant for 'train-val-test'")
    parser.add_argument("-of", "--n_outerfolds", type=int, default=3,
                        help="specify the number of outerfolds to use for 'nested_cv'"
                             "Standard is 5, only relevant for 'nested_cv'")
    parser.add_argument("-folds", "--n_innerfolds", type=int, default=5,
                        help="specify the number of innerfolds/folds to use for 'nested_cv' respectively 'cv-test'"
                             "Standard is 5, only relevant for 'nested_cv' and 'cv-test'")

    # Model and Optimization Params #
    parser.add_argument("-sf", "--save_final_model", default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="save the final model to hard drive "
                             "(caution: some models may use a lot of disk space, "
                             "unfitted models that can be retrained are already saved by default)")

    args = vars(parser.parse_args())

    model_reuse.retrain_on_new_data(**args)
