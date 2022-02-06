from setuptools import setup, find_packages

DISTNAME = 'jtb_2022_code'
VERSION = '0.1.0'
MAINTAINER = 'Chris Jackson'
MAINTAINER_EMAIL = 'cj59@nyu.edu'
LICENSE = 'MIT'

setup(name=DISTNAME,
      version=VERSION,
      author=MAINTAINER,
      author_email=MAINTAINER_EMAIL,
      license=LICENSE,
      packages=find_packages(include=['jtb_2022_code', "jtb_2022_code.*"]),
      install_requires=['numpy', 'scipy', 'scanpy', 'pandas', 'joblib', 'anndata'],
      zip_safe=True,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 4 - Beta"
      ]
     )
