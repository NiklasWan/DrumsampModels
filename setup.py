from setuptools import setup, find_packages

setup(
    name = 'drumsamp_models', 

    version='1.0', 

    packages=find_packages(),

    install_requires=[
        'tensorflow==2.5.0',
        'numpy==1.19.5',
        'scikit-learn==0.24.2',
        'librosa==0.7.0',
        'numba==0.48'
	],
    
    package_data={
        'drumsamp_models' : ['models/*.hdf5']
    }
)