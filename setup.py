#!/usr/bin/env python
"""networkx based analysis of continuous-time Markov chains on trees

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.
#
# More stuff added for Cython extensions.

from distutils.core import setup

from distutils.extension import Extension

from Cython.Distutils import build_ext

import numpy as np


setup(
        name='nxctmctree',
        version='0.1',
        description=DOCLINES[0],
        author='alex',
        url='https://github.com/argriffing/nxctmctree/',
        download_url='https://github.com/argriffing/nxctmctree/',
        packages=['nxctmctree'],
        test_suite='nose.collector',
        package_data={'nxctmctree' : ['tests/test_*.py']},
        cmdclass={'build_ext' : build_ext},
        ext_modules=[Extension('nxctmctree.rt', ['nxctmctree/rt.pyx'],
            include_dirs=[np.get_include()])],
        )
