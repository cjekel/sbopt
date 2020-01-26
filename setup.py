import io
from setuptools import setup

setup(
    name='sbopt',
    version='0.0.2',
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['sbopt'],
    url='https://github.com/cjekel/sbopt',
    license='MIT License',
    description='sbopt: Simple Surrogate-based optimization algorithm',
    long_description=io.open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 0.19.0",
        "pyDOE >= 0.3.8",
        "setuptools >= 38.6.0",
    ],
)
