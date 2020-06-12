# Copyright 2020 Larry Chen. All rights reserved.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PyPulse',
    version='0.1.0',
    author='Larry Chen',
    author_email='larrychen@berkeley.edu',
    description='Working with pulses in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['pypulse'],
    package_dir={'pypulse': 'pypulse'},
    provides=['pypulse'],
    install_requires=['numpy', 'matplotlib', 'qutip'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)