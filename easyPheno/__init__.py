import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import easyPheno.model as model
    import easyPheno.utils as utils
    import easyPheno.preprocess as preprocess
    import easyPheno.optimization as optimization
    import easyPheno.evaluation as evaluation
    from . import pipeline

__version__ = "0.1.0"
__author__ = 'Florian Haselbeck, Maura John, Dominik G. Grimm'
__credits__ = 'GrimmLab @ TUM Campus Straubing (https://bit.cs.tum.de/)'
