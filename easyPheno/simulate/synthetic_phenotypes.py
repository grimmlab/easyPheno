import numpy as np
import pandas as pd
import pathlib
import random

from ..preprocess import raw_data_functions
from ..utils import check_functions


def filter_duplicates(X: np.array, snp_ids: np.array) -> (np.array, np.array):
    """
    Remove duplicate SNPs, i.e. SNPs that are completely the same for all samples and therefore do not add information.

    :param X: genotype matrix to be filtered
    :param snp_ids: vector containing corresponding SNP ids

    :return: filtered genotype matrix and filtered SNP ids
    """
    uniques, index = np.unique(X, return_index=True, axis=1)
    X = uniques[:, np.argsort(index)]
    snp_ids = snp_ids[np.sort(index)]
    return X, snp_ids


def get_simulation(X: np.array, sample_ids: np.array, snp_ids: np.array, number_of_samples: int, number_causal_snps: int,
                   explained_variance: int, maf: int, heritability: int, seed: int,
                   number_background_snps: int, distribution: str, shape: float) \
                    -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array):
    """
    Simulate phenotypes based on (real) genotypes in an additive setting with normally distributed noise and normally or
    gamma distributed effect sized of causal SNPs.

    :param X: genotype matrix
    :param sample_ids: sample ids of genotype matrix
    :param snp_ids: SNP ids of genotype matrix
    :param number_of_samples: number of samples of synthetic phenotype
    :param number_causal_snps: number of SNPs used as causal markers in simulation
    :param explained_variance: percentage value of how much of the total variance the causal SNPs should explain
    :param maf: percentage value used for maf filtering of genotype matrix
    :param heritability: percentage value of how much of the variance should be explained by polygenic background
    :param seed: seed for random sampling
    :param number_background_snps: number of randomly selected SNPs to simulate the polygenic background
    :param distribution: probability distribution used to draw coefficients of causal SNPs can be 'normal' or 'gamma'
    :param shape: only needed if distribution is 'gamma'

    :return: simulated phenotype with corresponding sample ids, SNP ids of causal SNPs, SNP ids of background SNPs,
    effect sizes of background, effect sizes of causal SNPs, used explained variance for each causal SNP
    """

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
        raise Exception('Only %d SNPs are available after filtering. Cannot choose %d causal SNPs and %d background SNPs'
                        ' for simulations. Please check again'
                        % (len(snp_ids_sampled), number_causal_snps, number_background_snps))

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


def check_sim_id(sim_dir: pathlib.Path) -> int:
    """
    Check which ids were already used for simulations.

    :param sim_dir: directory containing simulations to check

    :return: last simulation number + 1
    """
    sim_ids = []
    for sim in sim_dir.iterdir():
        if sim.is_file() and 'Simulation_' in sim.as_posix():
            sim_numbers = sim.with_suffix('').name.split('_')[-1]
            sim_ids.append(int(sim_numbers.split('-')[-1]))
    if len(sim_ids) == 0:
        return 1
    else:
        return max(sim_ids) + 1


def save_sim_overview(save_dir: pathlib.Path, sim_names: list, number_of_samples: list, number_causal_snps: list,
                      explained_variance: list, maf: list, heritability: list, seeds: list,
                      number_background_snps: list, distribution: list, shape: list):
    """
    save overview file for all simulations; append new simulations if file already exists

    :param save_dir: directory to save overview file to
    :param sim_names: list containing simulation name for each simulation
    :param number_of_samples: list containing number of samples for each simulation
    :param number_causal_snps: list containing number of causal SNPS for each simulation
    :param explained_variance: list containing total explained variance of causal SNPs for each simulation
    :param maf: list containing used maf frequency for each simulation
    :param heritability: list containing used heritability for each simulation
    :param seeds: list containing used seed for each simulation
    :param number_background_snps: list containing number of background SNPs for each simulation
    :param distribution: list containing used distribution of causal effect sizes for each simulation
    :param shape: list containing shape of gamma distribution, resp. None for normal distribution for each simulation
    """
    overview_file = save_dir.joinpath('Simulations_Overview.csv')
    if not check_functions.check_exist_files([overview_file]):
        print('Create new overview file for simulations.')
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
        print('Overview file already exists. Will append new simulations to file.')
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
    """
    Set all variables, generate required simulations and save overview file and simulated phenotypes to save_dir and
    background SNPs, effect sizes/betas of background SNPs and config infos containing SNP ids and betas of causal SNPs
    to subfolder sim_configs, all with matching simulation ids as NAME_OF_FILE_sim_id.csv.
    If only one phenoype is simulated, the sim_id consists of a single number, if several phenotypes are simulated with
    the same configurations, then the sim_id is the number of the first simulation '-' number of last simulation,
    e.g. '10-15'

    :param save_dir: directory to save simulations to
    :param number_of_sim: number of simulations to create with same configurations
    :param X: genotype matrix
    :param sample_ids: sample ids of genotype matrix
    :param snp_ids: SNP ids of genotype matrix
    :param number_of_samples: number of samples of synthetic phenotype
    :param number_causal_snps: number of SNPs used as causal markers in simulation
    :param explained_variance: percentage value of how much of the total variance the causal SNPs should explain
    :param maf: percentage value used for maf filtering of genotype matrix
    :param heritability: percentage value of how much of the variance should be explained by polygenic background
    :param seed: seed for random sampling
    :param number_background_snps: number of randomly selected SNPs to simulate the polygenic background
    :param distribution: probability distribution used to draw coefficients of causal SNPs can be 'normal' or 'gamma'
    :param shape: only needed if distribution is 'gamma'
    """

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
          % (number_of_sim, number_of_samples, number_causal_snps, number_background_snps, heritability,
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
        new_seed = seed + sim_names[i]
        simulated_phenotype, sample_ids_sampled, causal_snps_ids, background_snp_ids, betas_background, beta, c = \
            get_simulation(X, sample_ids, snp_ids, number_of_samples, number_causal_snps, explained_variance, maf,
                           heritability, seed, number_background_snps, distribution, shape)

        causal_markers.append(causal_snps_ids)
        seeds.append(new_seed)
        background_markers.append(background_snp_ids)
        background_betas.append(betas_background)
        causative_beta.append(beta)
        ev.append(c)

        df_sim = pd.DataFrame({f'sim{sim_names[i]}': simulated_phenotype,
                               f'sim{sim_names[i]}_shift': simulated_phenotype + np.abs(
                                   np.min(simulated_phenotype)) + 1},
                              index=sample_ids_sampled)
        df_final = df_final.join(df_sim)

    # save overview
    save_sim_overview(save_dir=save_dir, sim_names=sim_names, number_of_samples=[number_of_samples] * number_of_sim,
                      number_causal_snps=[number_causal_snps] * number_of_sim,
                      explained_variance=[explained_variance] * number_of_sim, maf=[maf] * number_of_sim,
                      heritability=[heritability] * number_of_sim, seeds=seeds,
                      number_background_snps=[number_background_snps] * number_of_sim,
                      distribution=[distribution] * number_of_sim, shape=[shape] * number_of_sim)

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
