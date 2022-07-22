import argparse
import numpy as np
import pandas as pd
import pathlib
import random

from ..preprocess import raw_data_functions, encoding_functions
from ..utils import check_functions


def filter_duplicates(X: np.array, snp_ids: np.array) -> (np.array, np.array):
    uniques, index = np.unique(X, return_index=True, axis=1)
    X = uniques[:, np.argsort(index)]
    snp_ids = snp_ids[np.sort(index)]
    return X, snp_ids


def get_simulation(X: np.array, sample_ids: np.array, snp_ids: np.array, number_of_samples: int, number_causal_snps: int,
                   explained_variance: int, maf: int, heritability: int, seed: int,
                   number_background_snps: int, distribution: str, shape: float) \
                    -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array):
    # sanity checks
    if number_of_samples > len(sample_ids):
        print('Only %d samples are available. Cannot choose %d samples for simulations. Will use all available samples '
              'instead.' % (len(sample_ids), number_of_samples))
        number_of_samples = len(sample_ids)

    # get random samples for X
    random.seed(seed)
    np.random.seed(seed)
    samples_to_take = random.sample(list(enumerate(sample_ids)), number_of_samples)
    sample_indices = np.array(samples_to_take)[:, 0].astype(int)
    sample_ids_sampled = np.array(samples_to_take)[:, 1]
    X_sampled = X[sample_indices, :]

    # filter non-informative
    X_sampled, X_index = raw_data_functions.filter_non_informative_snps(X=X_sampled)
    snp_ids_sampled = snp_ids[X_index]

    # filter for duplicates
    X_sampled, snp_ids_sampled = filter_duplicates(X_sampled, snp_ids_sampled)

    # filter for MAF
    freq = raw_data_functions.get_minor_allele_freq(X=X_sampled)
    filter_indices = raw_data_functions.create_maf_filter(maf=maf, freq=freq)
    X_sampled = np.delete(X_sampled, filter_indices, axis=1)
    snp_ids_sampled = np.delete(snp_ids_sampled, filter_indices)

    # sanity checks
    if number_causal_snps > len(snp_ids_sampled):
        raise Exception('After filtering only %d SNPs remain. Not enough SNPs available to simulate %d causal SNPs.'
                        % (len(snp_ids_sampled), number_causal_snps))
    if number_causal_snps + number_background_snps > len(snp_ids_sampled):
        print('Only %d SNPs are available. Cannot choose %d causal SNPs and %d background SNPs for simulations. Will '
              'use %d SNPs as causal and remaining SNPs for background instead.'
              % (len(snp_ids_sampled), number_causal_snps, number_background_snps, number_causal_snps))
        number_background_snps = len(snp_ids_sampled) - number_causal_snps

    # compute simulations
    # choose random causal SNPs
    causal_snps = random.sample(list(enumerate(snp_ids_sampled)), number_causal_snps)
    causal_snps_indices = np.array(causal_snps)[:, 0].astype(int)
    causal_snps_ids = np.array(causal_snps)[:, 1]

    # choose background SNPs
    X_non_causal = np.delete(X_sampled, causal_snps_indices, axis=1)
    snp_ids_non_causal = np.delete(snp_ids_sampled, causal_snps_indices, axis=0)
    background_SNPs_indices = np.random.choice(X_non_causal.shape[1], number_background_snps, replace=False)
    background_SNPs = X_non_causal[:, background_SNPs_indices]
    background_snp_ids = snp_ids_non_causal[background_SNPs_indices]

    # compute effect size for background
    betas_background = np.random.normal(loc=0, scale=0.1, size=number_background_snps)

    # add background
    simulated_phenotype = np.matmul(background_SNPs, betas_background)

    # set heritability
    heritability = heritability / 100
    background_variance = np.var(simulated_phenotype)
    noise_variance = background_variance / heritability - background_variance

    # add random noise
    if distribution == 'gamma':
        random_noise = np.random.gamma(shape=shape, scale=np.sqrt(noise_variance / shape), size=number_of_samples)
    elif distribution == 'normal':
        random_noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=number_of_samples)
    else:
        raise Exception("Can only simulate noise with 'gamma' or 'normal' distribution.")
    simulated_phenotype = simulated_phenotype + random_noise

    # compute explained variances for more than 1 snp
    if number_causal_snps > 1:
        c = explained_variance
        mean = c / number_causal_snps
        sd = (mean / c) * 10
        explained_variance = np.random.normal(mean, sd, number_causal_snps)
        explained_variance.sort()
    else:
        explained_variance = np.array([explained_variance])
    explained_variance = explained_variance / 100

    # add causative markers with effect sizes
    caus_beta = []
    for i in range(number_causal_snps):
        beta = np.sqrt((explained_variance[i] / (1 - explained_variance[i]) *
                        (np.var(simulated_phenotype) / np.var(X_sampled[:, causal_snps_indices[i]]))))
        simulated_phenotype += beta * X_sampled[:, causal_snps_indices[i]]
        caus_beta.append(beta)

    return simulated_phenotype, sample_ids_sampled, causal_snps_ids, background_snp_ids, betas_background, caus_beta, \
           explained_variance


def check_sim_id(sim_dir: pathlib.Path) -> (int):
    sim_ids = []
    for sim in sim_dir.iterdir():
        if sim.is_file() and 'Simulation' in sim.as_posix():
            sim_numbers = sim.with_suffix('').name.split('_')[-1]
            sim_ids.append(int(sim_numbers.split('-')[-1]))
    if len(sim_ids) == 0:
        return 1
    else:
        return max(sim_ids) + 1


def save_sim_overview(save_dir: pathlib.Path, sim_names: list, number_of_samples: list, number_causal_snps: list,
                      explained_variance: list, maf: list, heritability: list, seeds: list,
                      number_background_snps: list, distribution: list, shape: list):

    overview_file = save_dir.joinpath('Simulations_Overview.csv')
    if not check_functions.check_exist_files([overview_file]):
        df_sim = pd.DataFrame({'simulation': sim_names,
                               'seed': seeds,
                               'heritability': heritability,
                               'MAF': maf,
                               'samples': number_of_samples,
                               'causal_SNPs': number_causal_snps,
                               'background_SNPs': number_background_snps,
                               'explained_var': explained_variance,
                               'distribution': distribution,
                               'shape': shape})
    else:
        df_old = pd.read_csv(overview_file)
        df_new = pd.DataFrame({'simulation': sim_names,
                               'seed': seeds,
                               'heritability': heritability,
                               'MAF': maf,
                               'samples': number_of_samples,
                               'causal_SNPs': number_causal_snps,
                               'background_SNPs': number_background_snps,
                               'explained_var': explained_variance,
                               'distribution': distribution,
                               'shape': shape})
        df_sim = pd.concat([df_old, df_new])
    df_sim.to_csv(overview_file, index=False)


def save_simulation(save_dir: pathlib.Path, number_of_sim: int, X: np.array, sample_ids: np.array, snp_ids: np.array,
                    number_of_samples: int, number_causal_snps: int, explained_variance: int, maf: int, heritability: int,
                    seed: int, number_background_snps: int, distribution: str, shape: float):
    # fix numbers / names of simulations
    sim_number = int(check_sim_id(sim_dir=save_dir))
    if number_of_sim > 1:
        sim_names = np.arange(sim_number, sim_number + number_of_sim)
        sim_id = str(sim_names[0]) + '-' + str(sim_names[-1])
    elif number_of_sim == 1:
        sim_names = [sim_number]
        sim_id = str(sim_number)
    else:
        raise Exception('number of simulations has to be at least 1')

    print('Now create %d simulations with %d samples, %d causal SNPs, %d background SNPs, '
          'heritability of %d, %d explained variance, %d maf and %s distribution'
          %(number_of_sim, number_of_samples, number_causal_snps, number_background_snps, heritability,
            explained_variance, maf, distribution))
    print('Save simulations with sim_id ' + sim_id)
    # create simulations
    causal_markers = []
    seeds = []
    background_markers = []
    background_betas = []
    causative_beta = []
    ev = []
    df_final = pd.DataFrame(index=sample_ids)
    for i in range(number_of_sim):
        seed = seed + sim_names[i]
        simulated_phenotype, sample_ids_sampled, causal_snps_ids, background_snp_ids, betas_background, beta, c = \
            get_simulation(X, sample_ids, snp_ids, number_of_samples, number_causal_snps, explained_variance, maf,
                           heritability, seed, number_background_snps, distribution, shape)

        causal_markers.append(causal_snps_ids)
        seeds.append(seed)
        background_markers.append(background_snp_ids)
        background_betas.append(betas_background)
        causative_beta.append(beta)
        ev.append(c)

        df_sim = pd.DataFrame({f'sim{sim_names[i]}': simulated_phenotype,
                               f'sim{sim_names[i]}_shift': simulated_phenotype + np.abs(
                                   np.min(simulated_phenotype)) + 1},
                              index=sample_ids_sampled)
        df_final = df_final.join(df_sim)

    # save simulations
    df_final.to_csv(save_dir.joinpath(f'Simulation_{sim_id}.csv'))
    # save configs
    df_causal = pd.DataFrame({'simulation': sim_names,
                              'seed': seeds,
                              'heritability': heritability,
                              'samples': number_of_samples,
                              'SNPs': number_causal_snps,
                              'explained_var': ev,
                              'causal_marker': causal_markers,
                              'causal_beta': causative_beta,
                              'distribution': distribution,
                              'shape': shape})

    df_causal.to_csv(save_dir.joinpath('sim_configs', f'simulation_config_{sim_id}.csv'), index=False)
    # save background markers and betas
    col = []
    for sim in sim_names:
        col.append('sim' + str(sim))
    col = tuple(col)
    bg = np.array(background_markers).T
    df_background = pd.DataFrame(bg, columns=col)
    df_background.to_csv(save_dir.joinpath('sim_configs', f'background_{sim_id}.csv'), index=False)
    bb = np.array(background_betas).T
    df_bb = pd.DataFrame(bb, columns=col)
    df_bb.to_csv(save_dir.joinpath('sim_configs', f'betas_background_{sim_id}.csv'), index=False)

    save_sim_overview(save_dir=save_dir, sim_names=sim_names, number_of_samples=[number_of_samples]*number_of_sim,
                      number_causal_snps=[number_causal_snps]*number_of_sim,
                      explained_variance=[explained_variance]*number_of_sim, maf=[maf]*number_of_sim,
                      heritability=[heritability]*number_of_sim, seeds=seeds,
                      number_background_snps=[number_background_snps]*number_of_sim,
                      distribution=[distribution]*number_of_sim, shape=[shape]*number_of_sim)


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
    parser.add_argument("-shape", type=float, default=1.0,
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
    print('Have genotype matrix %s.', args['genotype_matrix'])
    save_simulation(save_dir=geno_dir, number_of_sim=args['number_of_simulations'], X=X, sample_ids=sample_ids,
                    snp_ids=snp_ids, number_of_samples=args['number_of_samples'],
                    number_causal_snps=args['number_causal_snps'], explained_variance=args['explained_variance'],
                    maf=args['maf'], heritability=args['heritability'], seed=args['seed'],
                    number_background_snps=args['number_background_snps'], distribution=args['distribution'],
                    shape=args['shape'])
