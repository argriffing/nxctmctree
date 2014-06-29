#!/usr/bin/env python
"""networkx based analysis of continuous-time Markov chains on trees

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.

from distutils.core import setup

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
        )


