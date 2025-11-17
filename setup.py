from setuptools import setup, find_packages

setup(
    name='double-heston-calibration',
    version='1.0.0',
    description='Hybrid calibration system for Double Heston + Jump Diffusion model',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'tensorflow>=2.13.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
    ],
)
