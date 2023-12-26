from setuptools import setup, find_packages

setup(
    name='Dare',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.4', 'omegaconf==2.3.0', 'pandas==2.0.3', 'ray==2.8.1', 'scikit-learn==1.1.2', 'torch==1.12.1', 'ray==2.8.1', 'tqdm==4.50.2'
    ],
)