cdef extern from "hello.cpp":
    pass

cdef extern from "hello.h" namespace "greetings":
    cdef cppclass Hello:
        Hello() except +  # The last keyword allows cython to handle exceptions
        void printMessage()
