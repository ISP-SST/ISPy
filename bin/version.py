#!/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
from distutils.version import StrictVersion
from pathlib import Path

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

# Return the git revision as a string
def git_version(): # This function is not valid under PEP440

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




def get_version_info(): # This function is not valid under PEP440
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of ISPy.version messes up the build under Python 3.
    ISRELEASED = False
    if subprocess.call(['git', 'rev-parse']) == 0:
        GIT_REVISION, VERSION = git_version()
        if VERSION is None:
            VERSION = 'dev-' + GIT_REVISION[:7]
        else:
            ISRELEASED = True
    elif Path(__file__).parents[1].joinpath('ISPy','version.py').exists():
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



# This function is not valid under PEP440
def write_version_py(filename=str(Path(__file__).parents[1].joinpath('ISPy','version.py'))):
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





# Adding an alternative way compatible with PEP440:
def get_pep440version_info():
    # This version of the function uses the format {tag_name}.dev{num_commits}+{commit_hash} to
    # indicate that the version number is a development version, with {tag_name} indicating the
    # name of the latest tag, {num_commits} representing the number of commits since the tag, and
    # {commit_hash} representing the abbreviated commit hash.
    git_describe_output = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
    
    # Split the output string
    parts = git_describe_output.split('-')
    if len(parts) == 1:
        # No additional commits since the last tag, so this is a release version
        return parts[0]
    else:
        # There are additional commits since the last tag, creating a development version
        tag_name, num_commits, commit_hash = parts
        commit_hash = commit_hash[1:]  # Remove the 'g' prefix from the commit hash
        return f"{tag_name}.dev{num_commits}+{commit_hash}"
    