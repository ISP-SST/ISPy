import os
import sys
import subprocess
import numpy
#from setuptools import find_packages
from distutils.core import setup, Extension, Command
import shutil
from glob import glob
import shlex

from bin.generate_module_list import generate_module_list
from bin.version import *

# This directory
dir_setup = os.path.dirname(os.path.realpath(__file__))

def generate_cython():
	cwd = os.path.abspath(os.path.dirname(__file__))
	print("Cythonizing sources")
	for d in generate_module_list():
            p = subprocess.call([sys.executable, os.path.join(cwd, 'bin', 'cythonize.py'), os.path.join(*'{0}'.format(d).split('.'))], cwd=cwd)
            if p != 0:
                raise RuntimeError("Running cythonize failed!")

class clean(Command):
    """
    Cleans *.pyc and cython source files.
    """

    description  = "remove build files"
    user_options = [("all", "a", "the same")]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        curr_dir = os.getcwd()

        patterns = ["*~", ".*~"]
        for root, dirs, files in os.walk(dir_setup):
            for file in files:
                if file.endswith('.pyc'):
                    os.remove(os.path.join(root, file))
                if file.endswith('.pyx'):
                    cfile = os.path.join(root, os.path.splitext(file)[0]+'.c')
                    if os.path.isfile(cfile):
                        os.remove(cfile)
            for dir in dirs:
                for pat in patterns:
                    for f in glob(os.path.join(root, dir, pat)):
                        os.remove(f)
            for pat in patterns:
                for f in glob(os.path.join(root, pat)):
                    os.remove(f)

        os.chdir(dir_setup)
        names = ["MANIFEST", "build", "dist", "cythonize.dat", "ISPy/version.py"]

        for f in names:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)

        os.chdir(curr_dir)

def setup_package():
    # Rewrite the version file everytime
    write_version_py()

    if "--cythonize" in sys.argv:
        sys.argv.remove("--cythonize")
        # Generate Cython sources, unless we're generating an sdist
        generate_cython()

    with open("README.md", "r") as fh:
        long_description = fh.read()

    # Find and prepare extension modules
    ext_modules  = []
    for m in generate_module_list(with_ext=True):
        modpath = os.path.join(*m.split('.'))
        loc = {}
        execfile(os.path.join(modpath, '__extensions__.ispy'), loc)
        extmods = loc['ext_modules']
        for e in extmods:
            e.name = m+'.'+e.name
            e.sources = [s if os.path.isabs(s) else os.path.join(modpath,s) for s in e.sources]
            e.include_dirs = [s if os.path.isabs(s) else os.path.join(modpath,s) for s in e.include_dirs]
            e.library_dirs = [s if os.path.isabs(s) else os.path.join(modpath,s) for s in e.library_dirs]
            e.runtime_library_dirs = [s if os.path.isabs(s) else os.path.join(modpath,s) for s in e.runtime_library_dirs]
        ext_modules.extend(extmods)

    # Gather package data from all modules
    package_data = {}
    for m in generate_module_list():
        package_data[m] = []
        packdatfile = os.path.join(*(m.split('.')+['package_data']))
        if os.path.isfile(packdatfile):
            with open(packdatfile, 'rt') as f:
                for l in f.readlines():
                    package_data[m].extend(shlex.split(l, comments=True))

    setup(
        name                          = "ISPy",
        version                       = get_version_info()[0],
        author                        = "ISP-SST",
        author_email                  = "hillberg@astro.su.se",
        description                   = "Commonly used tools at the ISP",
        long_description              = long_description,
        long_description_content_type = "text/markdown",
        url                           = "https://github.com/ISP-SST/ISPy",
        packages                      = generate_module_list(),
        package_data                  = package_data,
        include_package_data          = True,
        classifiers                   = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires               = '>=2.7',
        ext_modules                   = ext_modules,
	cmdclass                      = {
	    'clean' : clean
	}
    )

if __name__ == '__main__':
    setup_package()
