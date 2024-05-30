from setuptools import setup
from Cython.Build import cythonize

setup(name = "Test",
      ext_modules = cythonize("*.pyx", annotate=True))
