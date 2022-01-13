import os


def get_list_of_implemented_models():
    """
    Create a list of all implemented models based on files existing in 'model' subdirectory of the repository
    """
    # Assumption: naming of python source file is the same as the model name specified by the user
    model_src_files = os.listdir('model')
    model_src_files.remove('__init__.py')
    model_src_files.remove('base_model.py')
    return [model[:-3] for model in model_src_files]
