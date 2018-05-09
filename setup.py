from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name="fplot",
    version="0.2.3",
    packages=find_packages(),

    install_requires=['numpy',
                      'matplotlib'],

    author="David O'Connor",
    author_email="david.alan.oconnor@gmail.com",
    url='https://github.com/David-OConnor/fplot',
    description="Plot functions with simple syntax",
    long_description=readme,
    license="Apache 2.0",
    keywords="plot, matplotlib, plotting, functions, jupyter",
)
