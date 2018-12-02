from setuptools import setup
import os

with open("README.md", "r") as f:
    long_description = f.read()

__PATH__ = os.path.dirname(os.path.realpath(__file__))

setup(
    name='olympic',
    version='0.1',
    description='A port of the Keras API to PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/oscarknagg/olympic-pytorch',
    author='Oscar Knagg',
    author_email='oscar@knagg.co.uk',
    license='MIT',
    packages=['olympic'],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',  # Untested on other python versions
        'License :: OSI Approved :: MIT License',
        "Operating System :: Unix",  # Untested on other OS
    ]
)
