from setuptools import setup, find_packages

setup(

    name='hdb5_io_tools',

    version='1.0.0',

    packages=find_packages(),

    author='D-ICE Engineering',

    author_email='',

    description='Python module for manipulating hydrodynamic database (HDB5-IO)',

    long_description=open('README.rst').read(),

    include_package_data=True,

    install_requires=['numpy',
                      'matplotlib',
                      'h5py',
                      'argcomplete',
                      "meshmagick@git+ssh://git@d-ice.gitlab.host:/common/hydro/meshmagick.git@develop"],

    entry_points={
        'console_scripts': ['hdb5tool=hdb5_io.HDB5tool.HDB5tool:main',
        'hdb5merge=hdb5_io.HDB5merge.HDB5merge:main']
    },

    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Development Status :: Production',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],

)
