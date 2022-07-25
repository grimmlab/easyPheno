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
                        help="Specify the name (including data type suffix) of the genotype matrix to be used. "
                             "Needs to be located in the specified data_dir."
                             "For more info regarding the required format see our documentation.")
    parser.add_argument("-nsim", "--number_of_simulations", type=int, default=1,
                        help="Specify the number of simulations to create with the same configurations.")
    parser.add_argument("-nsamp", "--number_of_samples", type=int, default=1000,
                        help="Specify the number of samples the simulated phenotype should have. The maximum is the "
                             "number of samples of the genotype matrix")
    parser.add_argument("-ncaus", "--number_causal_snps", type=int, default=1,
                        help="Specify the number of causal SNPs for the simulation. Need at least one causal SNP.")
    parser.add_argument("-nback", "--number_background_snps", type=int, default=1000,
                        help="Specify the number of SNPs used to simulate the polygenic background. Need at least one "
                             "SNP.")
    parser.add_argument("-ev", "--explained_variance", type=int, default=30,
                        help="Specify the total explained variance of the causal SNPs, i.e. how much of the total "
                             "variance of the phenotype should be explained by the causal SNPs.")
    parser.add_argument("-maf", type=int, default=0,
                        help="Specify the minor allele frequency (as percentage value). "
                             "specify 0 if you do not want a maf filter.")
    parser.add_argument("-her", "--heritability", type=int, default=70,
                        help="Specify the heritability of the phenotype, i.e. how much of the variance of "
                             "polygenic background + noise should be explained by the polygenic background.")
    parser.add_argument("-seed", type=int, default=42,
                        help="Specify the seed for random sampling.")
    parser.add_argument("-dist", "--distribution", type=str, default="normal",
                        help="Specify the distribution of the noise. Can be 'gamma' or 'normal'")
    parser.add_argument("-shape", type=float, default=None,
                        help="If distribution is 'gamma', specify the shape parameter of the distribution. If not "
                             "specified it will be set to 1.0")
    parser.add_argument("-sd", "--save_dir", type=str, default=None,
                        help="Define save directory for the synthetic phenotypes. Default is the data directory.")
    args = vars(parser.parse_args())
    data_dir = pathlib.Path(args['data_dir'])
    if args['save_dir'] is None:
        args['save_dir'] = args['data_dir']
    if args['distribution'] == 'gamma' and args['shape'] is None:
        args['shape'] = 1.0

    # load genotype
    check_functions.check_exist_directories(list_of_dirs=[data_dir], create_if_not_exist=False)
    check_functions.check_exist_files([data_dir.joinpath(args['genotype_matrix'])])
    X, sample_ids, snp_ids = raw_data_functions.check_transform_format_genotype_matrix(data_dir=data_dir,
                            genotype_matrix_name=args['genotype_matrix'], models=None, user_encoding='012')
    X = encoding_functions.get_additive_encoding(X)
    print('Have genotype matrix ', args['genotype_matrix'])

    # create and save simulations
    save_simulation(save_dir=args['save_dir'], genotype_matrix_name=args['genotype_matrix'],
                    number_of_sim=args['number_of_simulations'], X=X, sample_ids=sample_ids,
                    snp_ids=snp_ids, number_of_samples=args['number_of_samples'],
                    number_causal_snps=args['number_causal_snps'], explained_variance=args['explained_variance'],
                    maf=args['maf'], heritability=args['heritability'], seed=args['seed'],
                    number_background_snps=args['number_background_snps'], distribution=args['distribution'],
                    shape=args['shape'])
