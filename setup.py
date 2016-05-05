# -*- coding: utf-8 -*-
# on bigedj: install to /usr/local/lib/python2.7/dist-packages
from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy

setup(
    name = "pl",
    author = "Scott Brown",
    author_email = "sbrown103@gmail.com",
    version = "1.0",
    packages = ["pl"],
    include_dirs = [cython_gsl.get_include(), numpy.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("pl.regression",
                             ["pl/regression.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir()])]
    )