from setuptools import setup, find_packages


setup(
    name='dd_vae',
    packages=find_packages(),
    python_requires='>=3.5.0',
    version='0.1',
    install_requires=[
        'tqdm', 'numpy',
        'pandas', 'scipy',
        'torch', 'networkx',
        'Theano'
    ],
    description=('DD-VAE'),
)
