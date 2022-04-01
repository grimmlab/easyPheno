import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import easyPheno.evaluation as evaluation
    import easyPheno.model as model
    import easyPheno.utils as utils
    import easyPheno.optimization as optimization
    import easyPheno.preprocess as preprocess

    from . import optim_pipeline

__version__ = "0.1.8"
__author__ = 'Florian Haselbeck, Maura John, Dominik G. Grimm'
__credits__ = 'GrimmLab @ TUM Campus Straubing (https://bit.cs.tum.de/)'
