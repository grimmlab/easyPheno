from setuptools import setup
from setuptools import find_packages
import os


def get_long_description():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''


def get_version():
    with open('easyPheno/__init__.py') as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]


def get_required():
    try:
        with open('Docker/requirements.txt') as f:
            return f.read().splitlines()
    except:
        print('Problems with reading the requirements.txt file')


setup(
    # Project Information
    name='easypheno',
    version=get_version(),
    author='Florian Haselbeck; Maura John, Dominik G. Grimm',
    author_email='florian.haselbeck@tum.de',
    license='MIT',
    # Short and long description (sing readme)
    description='Easy-to-use state-of-the-art phenotype prediction framework',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    # URLs
    url='https://github.com/grimmlab/easyPheno',
    project_urls={
        "Documentation": "https://easypheno.readthedocs.io/"
    },
    # build stuff
    packages=find_packages(),
    install_requires=get_required(),

    # see https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    python_requires=">=3.8"
)
