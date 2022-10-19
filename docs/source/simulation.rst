Synthetic data
========================================
I this tutorial we will show you how to use easyPheno to create synthetic phenotypes for real genotypes.

Besides the written tutorial, we recorded a :ref:`Video tutorial: Synthetic data generation`, which is embedded below.

Additive model
"""""""""""""""
To create synthetic phenotypes, easyPheno uses an additive model

    .. math::

        y = X \beta + Z \gamma + \epsilon

where the phenotype :math:`y` is given as the sum of one or more causal markers :math:`X`
with effect sizes :math:`\beta`; random effects :math:`Z` with small effect sizes :math:`\gamma` drawn from a Gaussian
distribution, which simulate the polygenic background; and some noise :math:`\epsilon`.

Create synthetic data in easyPheno
""""""""""""""""""""""""""""""""""""""""
To create a synthetic phenotype all you need is the path to the folder where your data is stored (``data_dir``)
and the name of your genotype matrix (``name_of_genotype_matrix``).
Please read our :ref:`Data Guide` for more information on the data structure of the genotype matrix.

    .. code-block::

        python3 -m easypheno.simulate.run_synthetic_phenotypes --data_dir data_dir --genotype_matrix name_of_genotype_matrix

This will create a subfolder ``name_of_genotype_matrix`` within the ``data_dir`` and save two files,
where each simulation gets a unique number or ID (sim_id) to distinguish them from each other:

:Simulation_{sim_id}.csv: Contains the sample IDs corresponding to the genotype matrix, a column for the simulated phenotype (e.g. ``sim1``) and one column with the same phenotype but shifted to get rid of negative values (``sim1_shift``)
:Simulations_Overview.csv: Contains the sim_id and additional information such as number of samples, number of causal SNPs, etc. for each simulation

And within another subfolder ``sim_configs`` three files containing additional information:

:simulation_config_{sim_id}.csv: Contains detailed information of the phenotype such as the SNP ID and effect size of causal markers.
:background_{sim_id}.csv: Contains all SNP IDs of the used background markers
:betas_background_{sim_id}.csv: Contains the effect size for each background marker in the same order as the background SNPs

Per default easyPheno creates synthetic phenotypes with 1000 samples, and 1000 markers to simulate the polygenic
background with a heritability of 70%, i.e. such that the background accounts for 70% of the phenotypic variance.
To change that you can specify the number of samples (``--number_of_samples``), number of background markers
(``--number_background_snps``) and heritability (``--heritability``). For example

    .. code-block::

        python3 -m easypheno.simulate.run_synthetic_phenotypes --data_dir data_dir --genotype_matrix name_of_genotype_matrix --number_of_samples 100 --number_background_snps 200 --heritability 50

will create a phenotype with 100 samples and use 200 markers to simulate the background with a heritability of 50%.

easyPheno will use one causal marker for the synthetic phenotypes that explains 30% of the total variance. You can
adjust that by specifying the number of causal markers (``--number_causal_snps``) and the explained variance
(``--explained_variance``). For example

    .. code-block::

        python3 -m easypheno.simulate.run_synthetic_phenotypes --data_dir data_dir --genotype_matrix name_of_genotype_matrix --number_causal_snps 5 --explained_variance 20

will create a phenotype with 5 causal markers that together explain around 20% of the total phenotypic variance.

It is also possible to simulate phenotypes with a skewed distribution by using the flag ``--distribution 'gamma'``.
If you use a gamma distribution you can additionally adjust the shape parameter with ``-shape``.

If you want to create several phenotypes with the same specifications at once, you can specify the number of simulations
with ``--number_of_simulations``. Then the corresponding sim_id will contain the number of the first and last simulation,
e.g. '10-15' for the six simulations '10', '11', '12', '13', '14', '15'.

To get an overview over the other options you can adjust when creating synthetic phenotypes with easyPheno,
just use:

    .. code-block::

        python3 -m easypheno.simulate.run_synthetic_phenotypes --help

Video tutorial: Synthetic data generation
""""""""""""""""""""""""""""""""""""""""""""""

