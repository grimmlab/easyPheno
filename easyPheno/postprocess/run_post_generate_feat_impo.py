import argparse

from . import feat_importance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data_dir", type=str,
                        help="Provide the full path of your data directory (that contains the geno- and phenotype "
                             "files as well as the index file).")
    parser.add_argument("-rd", "--results_dir", type=str,
                        help="Provide the full path of the directory where your results are stored and for which "
                             "you want to post-generate feature importances")
    args = vars(parser.parse_args())
    data_dir = args['data_dir']
    results_directory_genotype_level = args['results_dir']

    feat_importance.post_generate_feature_importances(
        results_directory_genotype_level=results_directory_genotype_level, data_dir=data_dir
    )
