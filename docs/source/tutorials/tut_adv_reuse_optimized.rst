HowTo: Reuse optimized model
==================================================
easyPheno enables the reuse of an optimized model for two use cases:

- Run inference using final model on new data (SNP ids of old and new data (after preprocessing, e.g. MAF and duplicate filtering) have to match exactly)

- Retrain on new data using best hyperparameter combination of a previous optimization run

We provide scripts to run these functions (prefix *run_*) with our :ref:`Docker workflow`, on which we will also focus
in this tutorial. If you want to use the functions directly (e.g. with the pip installed package),
please check the scripts and see which functions are called.

Run inference on new data
""""""""""""""""""""""""""""""""""""""""""
The main use case for this is that you get new samples for the same phenotype, which come with the same SNP information,
and you want to apply a previously optimized model on them. If the final model was not saved, easyPheno will trigger a retraining
using the found hyperparameters and old dataset. To apply this prediction model on new data, the SNPs of the old and new dataset need to match exactly!

To apply a final prediction model on new data, you have to run the following command:

    .. code-block::

        python3 -m easypheno.postprocess.run_inference -rd full_path_to_model_results -odd path_to_old_data -ndd path_to_new_data -ngm name_new_genotype_matrix -npm name_new_phenotype_matrix -sd path_to_save_directory

By doing so, a .csv file containing the predictions on the whole new dataset will be created by applying the final prediction model, eventually after retraining on the old dataset if the final model was not saved.


Retrain on new data
""""""""""""""""""""""""""""
If you want to train a new prediction model with the same hyperparameters found for another dataset, you can call
`easypheno.postprocess.run_retrain_on_new_data <https://github.com/grimmlab/easyPheno/blob/b9b5d5e588f4201f84eca8617601081e8d034f92/easypheno/postprocess/run_retrain_on_new_data.py>`_.
As this script contains many parameters, we refer to the API documentation respective source code for more information.
Similar to applying a final model on new samples, a .csv file with the predictions on that new data will be created.
