HowTo: Summarize prediction results with easyPheno
======================================================
In the subpackage `postprocess <https://github.com/grimmlab/easyPheno/tree/main/easypheno/postprocess>`_, we included
functions to analyze optimization results. We provide scripts to run each of these functions (prefix *run_*) with our :ref:`Docker Workflow`, on which we will also focus
in this tutorial. If you want to use the functions directly (e.g. with the pip installed package),
please check the scripts and see which functions are called.

Optimization results in easyPheno are saved using the following directory structure: *user_defined_save_dir/results/name_genotype_matrix/name_phenotyp_matrix/name_phenotype/*
By running *run_summarize_results.py*, you can accumulate all optimization results for a genotype matrix:

    .. code-block::

        python3 -m easypheno.postprocess.run_summarize_results -rd path_at_name_genotype_matrix_level

This leads to the creation of the summary files described in `summarize_results_per_phenotype_and_datasplit() <https://github.com/grimmlab/easyPheno/blob/main/easypheno/postprocess/results_analysis.py#L10>`_.

Using a *Results_summary_all_phenotypes*DATASPLIT-PATTERN*.csv* file created by the command above, we provide scripts to visualize the results of several prediction models on different phenotypes:

    .. code-block::

        python3 -m easypheno.postprocess.run_plot_results -rsp path_to_Results_summary_all_phenotypes_XX.csv -sd path_to_save_directory

This creates a heatmap plot, which is stored at the specified save directory. Currently, heatmaps are implemented, and we can easily add more plot functions.


Additional analysis for simulated phenotypes
""""""""""""""""""""""""""""""""""""""""""""""
In addition, the subpackage `simulate <https://github.com/grimmlab/easyPheno/tree/main/easypheno/simulate>`_ contains
results analysis functions, which are only applicable for our simulated phenotypes (see :ref:`HowTo: Create synthetic phenotype data`).

For simulated phenotypes, we know the ground truth in terms of markers respective features, which influence the phenotypic value.
Based on that, we are able to compare these effect sizes with feature importances to analyze how well an algorithm captures the relevant features.

To this end, we conduct a statistical as well as visual analysis, which we further describe in the following publication:

    | **A comparison of classical and machine learning-based phenotype prediction methods on simulated data and three plant species**
    | Maura John, Florian Haselbeck, Rupashree Dass, Christoph Malisi, Patrizia Ricca, Christian Dreischer, Sebastian J. Schultheiss and Dominik G. Grimm
    | *Frontiers in Plant Science, 2022 (currently in press)*

The files to do that can be generated with the following command:

    .. code-block::

        python3 -m easypheno.simulate.run_results_analysis_synthetic_data -rd path_at_name_genotype_matrix_level -simd path_to_simulation_configs -sd path_to_save_directory

Besides .csv-files with statistical information, a scatter plot visualizing feature importances in comparison with effect sizes is created.