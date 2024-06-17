from setuptools import setup, find_packages

DISTNAME = 'jtb_2023_code'
VERSION = '1.0.0'
MAINTAINER = 'Chris Jackson'
MAINTAINER_EMAIL = 'cj59@nyu.edu'
LICENSE = 'MIT'

setup(
    name=DISTNAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    license=LICENSE,
    packages=find_packages(include=['jtb_2023_code', "jtb_2023_code.*"]),
    install_requires=[
      'numpy',
      'scipy',
      'scanpy',
      'pandas',
      'joblib',
      'anndata',
      'matplotlib',
      'supirfactor-dynamical',
      'inferelator-velocity',
      'pydeseq'
    ],
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ]
)
