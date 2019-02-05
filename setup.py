from setuptools import setup, find_packages

setup(
    name='oaprogression',
    version='0.1',
    author='Aleksei Tiulpin',
    author_email='aleksei.tiulpin@oulu.fi',
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE',
    long_description=open('README.md').read(),
)
