# distutils: language = c++

from hello cimport Hello as cHello

cdef class Hello:
    cdef cHello c_helloInst

    def __cinit__(self):
        self.c_helloInst = cHello()
    def printMessage(self):
        self.c_helloInst.printMessage()

def hello():
    helloInst = Hello()
    helloInst.printMessage()
