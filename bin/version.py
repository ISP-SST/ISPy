#!/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
from distutils.version import StrictVersion

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except (subprocess.CalledProcessError, OSError):
        GIT_REVISION = "Unknown"

    GIT_VERSION_TAG = None
    try:
        out = _minimal_ext_cmd(['git', 'tag', '--points-at'])
        tags = out.strip().decode('ascii').split()
        for tag in tags:
            if tag[0] == 'v':
                try:
                    StrictVersion(tag[1:])
                    GIT_VERSION_TAG = tag[1:]
                    break
                except:
                    pass
    except (subprocess.CalledProcessError, OSError):
        pass

    if not GIT_REVISION:
        # this shouldn't happen but apparently can
        GIT_REVISION = "Unknown"

    return GIT_REVISION, GIT_VERSION_TAG

def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of ISPy.version messes up the build under Python 3.
    ISRELEASED = False
    if os.path.exists('.git'):
        GIT_REVISION, VERSION = git_version()
        if VERSION is None:
            VERSION = 'dev-' + GIT_REVISION[:7]
        else:
            ISRELEASED = True
    elif os.path.exists('ISPy/version.py'):
        # must be a source distribution, use existing version file
        try:
            from numpy.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "ISPy/version.py and the build directory "
                              "before building.")
        try:
            from numpy.version import version      as VERSION
        except ImportError:
            raise ImportError("Unable to import version. Try removing "
                              "ISPy/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    return VERSION, GIT_REVISION, ISRELEASED

def write_version_py(filename='ISPy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM ISPY SETUP.PY
#
# To compare versions robustly, use `ISPy.lib.IspyVersion`
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
"""
    VERSION, GIT_REVISION, ISRELEASED = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': VERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()
