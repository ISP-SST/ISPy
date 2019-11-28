#!/bin/env python
# -*- coding: utf-8 -*-

from os.path import dirname, join
from sys import stdout

def hello():
    with open(join(dirname(__file__), 'hello.txt')) as f:
        stdout.write(f.read())

if __name__ == '__main__':
    hello()
