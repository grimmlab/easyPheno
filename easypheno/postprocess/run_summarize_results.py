import argparse

from . import results_analysis

if __name__ == "__main__":
    """
    Run to gather some overview files on the optimization results for the specified results directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--results_dir", type=str,
                        help="Provide the full path of the directory where your results are stored "
                             "(name of the genotype matrix level)")
    parser.add_argument("-evm", "--eval_metric", type=str,
                        help="Eval metric to use for results summary. Options, default given first: "
                             "Regression: explained_variance, r2_score, rmse, mse. "
                             "Classification: mcc, f1_score, accuracy, precision, recall.")
    args = vars(parser.parse_args())
    results_directory_genotype_level = args['results_dir']
    eval_metric = args['eval_metric']

    results_analysis.summarize_results_per_phenotype_and_datasplit(
        results_directory_genotype_level=results_directory_genotype_level, eval_metric=eval_metric
    )
