import argparse

from . import model_reuse

if __name__ == "__main__":
    """
    Run to apply the specified model on a dataset containing new samples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-odd", "--old_data_dir", type=str, default=None,
                        help="Provide the full path of the old data directory (that contains the geno- and phenotype "
                             "files as well as the index file the model was trained on).")
    parser.add_argument("-ndd", "--new_data_dir", type=str,
                        help="Provide the full path of the new data directory that contains the geno- and phenotype "
                             "files you want to predict on")
    parser.add_argument("-ngm", "--new_genotype_matrix", type=str,
                        help="Provide the name of the new genotype matrix you want to predict on")
    parser.add_argument("-npm", "--new_phenotype_matrix", type=str, default=None,
                        help="Optional: Provide the name of the new phenotype matrix you want to predict on - if available - to directly get metrics of your prediction model.")
    parser.add_argument("-sd", "--save_dir", type=str,
                        help="Define the save directory for the results.")
    parser.add_argument("-rd", "--results_dir_model", type=str,
                        help="Provide the full path of the directory where your results of the model "
                             "you want to use are stored")
    args = vars(parser.parse_args())
    old_data_dir = args['old_data_dir']
    new_data_dir = args['new_data_dir']
    new_genotype_matrix = args['new_genotype_matrix']
    new_phenotype_matrix = args['new_phenotype_matrix']
    save_dir = args["save_dir"]
    results_directory_model = args['results_dir_model']

    model_reuse.apply_final_model(
        results_directory_model=results_directory_model, old_data_dir=old_data_dir, new_data_dir=new_data_dir,
        new_genotype_matrix=new_genotype_matrix, new_phenotype_matrix=new_phenotype_matrix, save_dir=save_dir
    )
