import argparse

from.results_analysis_synthetic_data import featimps_vs_simulation

if __name__ == "__main__":
    """
    Run file to generate statistics and plots of feature importances versus effect sizes on synthetic data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--results_dir", type=str,
                        help="Provide the full path of the directory where your results are stored "
                             "(name of the genotype matrix level)")
    parser.add_argument("-simd", "--sim_config_dir", type=str,
                        help="Provide the full path of the directory where the simulation configs are stored")
    parser.add_argument("-sd", "--save_dir", type=str, default=None,
                        help="Define save directory for the plots. Default is the same as results directory.")

    args = vars(parser.parse_args())
    results_directory_genotype_level = args['results_dir']
    sim_config_dir = args['sim_config_dir']
    save_dir = args['save_dir']

    featimps_vs_simulation(
        results_directory_genotype_level=results_directory_genotype_level,
        sim_config_dir=sim_config_dir, save_dir=save_dir
    )
