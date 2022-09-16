HowTo: Summarize prediction results with easyPheno
======================================================
In the subpackage `postprocess <https://github.com/grimmlab/easyPheno/tree/main/easypheno/postprocess>`_, we included
functions to analyze optimization results. We provide scripts to run each of these functions (prefix *run_*) on which we will also focus
in this tutorial. If you want to use the functions directly (e.g. with the pip installed package),
please check the scripts and see which functions are called.

Optimization results in easyPheno are saved using the following directory structure: *user_defined_save_dir/results/name_genotype_matrix/name_phenotyp_matrix/name_phenotype/*
By running *run_summarize_results.py*, you can accumulate all optimization results for a genotype matrix:

    .. code-block::

        python3 -m easypheno.postprocess.run_summarize_results -rd path_at_name_genotype_matrix_level

files die dadurch entstehen kurz beschreiben

plot results + output







Furthermore, the subpackage `simulate <https://github.com/grimmlab/easyPheno/tree/main/easypheno/simulate>`_ contains
results analysis functions, which are only applicable for our simulated phenotypes (see :ref:`HowTo: Create synthetic phenotype data`).

featimps code zeigen

outcome zeigen

