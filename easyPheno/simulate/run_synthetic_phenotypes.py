import argparse
import pathlib

from ..preprocess import raw_data_functions, encoding_functions
from ..utils import check_functions
from .synthetic_phenotypes import save_simulation


if __name__ == "__main__":
    """
    Run file to generate synthetic phenotypes
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data_dir", type=str,
                        help="Provide the full path of the directory where your genotype data is stored")
    parser.add_argument("-gm", "--genotype_matrix", type=str,
                        help="specify the name (including data type suffix) of the genotype matrix to be used. "
                             "Needs to be located in the specified data_dir."
                             "For more info regarding the required format see our documentation.")
    parser.add_argument("-nsim", "--number_of_simulations", type=int, default=1,
                        help="")
    parser.add_argument("-nsamp", "--number_of_samples", type=int, default=1000,
                        help="")
    parser.add_argument("-ncaus", "--number_causal_snps", type=int, default=1,
                        help="")
    parser.add_argument("-nback", "--number_background_snps", type=int, default=1000,
                        help="")
    parser.add_argument("-ev", "--explained_variance", type=int, default=30,
                        help="")
    parser.add_argument("-maf", type=int, default=0,
                        help="")
    parser.add_argument("-her", "--heritability", type=int, default=70,
                        help="")
    parser.add_argument("-seed", type=int, default=42,
                        help="")
    parser.add_argument("-dist", "--distribution", type=str, default="normal",
                        help="")
    parser.add_argument("-shape", type=float, default=None,
                        help="")
    parser.add_argument("-sd", "--save_dir", type=str, default=None,
                        help="Define save directory for the synthetic phenotypes. Default is the data directory.")
    args = vars(parser.parse_args())
    data_dir = pathlib.Path(args['data_dir'])
    if args['save_dir'] is None:
        save_dir = args['data_dir']
    else:
        save_dir = pathlib.Path(args['save_dir'])
    geno_dir = save_dir.joinpath(args['genotype_matrix']).with_suffix('')
    sim_config_dir = geno_dir.joinpath('sim_configs')
    check_functions.check_exist_directories(list_of_dirs=[data_dir, save_dir, geno_dir, sim_config_dir],
                                            create_if_not_exist=True)
    check_functions.check_exist_files([save_dir.joinpath(args['genotype_matrix'])])
    X, sample_ids, snp_ids = raw_data_functions.check_transform_format_genotype_matrix(data_dir=data_dir,
                            genotype_matrix_name=args['genotype_matrix'], models=None, user_encoding='012')
    X = encoding_functions.get_additive_encoding(X)
    print('Have genotype matrix ', args['genotype_matrix'])
    save_simulation(save_dir=geno_dir, number_of_sim=args['number_of_simulations'], X=X, sample_ids=sample_ids,
                    snp_ids=snp_ids, number_of_samples=args['number_of_samples'],
                    number_causal_snps=args['number_causal_snps'], explained_variance=args['explained_variance'],
                    maf=args['maf'], heritability=args['heritability'], seed=args['seed'],
                    number_background_snps=args['number_background_snps'], distribution=args['distribution'],
                    shape=args['shape'])
