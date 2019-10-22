"""
Python tools for I/O to binary files with named variables
Written by J. de la Cruz Rodriguez (ISP-SU 2019)

The idea is to write variables, one after the other in a binary file.
The file has a 5 byte header at the beginning. Then variables are written as follow:

Variable header:
1 int32 tell the routines the size of the header,
50 bytes contain the variable name, 1 int32 contain the number of dimensions, 
1 int32 contains the dataype, ndim*int64 contain the dimensions of the variable,
and then comes the raw data.

"""

import numpy as np
import os.path
#import ipdb 
import copy

# ==================================================================================

class container:
    def __init(self):
        pass

# ==================================================================================

VSIZE = 50
FSIZE = 5

# ==================================================================================

def file_exists(fname):
    try:
        kk = open(fname, 'r')
        kk.close()
        return True
    except:
        return False

    
# ==================================================================================

def get_header_size( ndim):
    longsize=8
    intsize=4
    return VSIZE + 2*intsize + ndim*longsize

# ==================================================================================

def create_header(vname, dim, dtype='float32'):
    hsize = get_header_size(len(dim))
    head = bytearray(hsize)
    
    # Write variable name to buffer
    vlen = len(vname)
    vname_byte = bytearray(VSIZE)
    vnamenull = vname+'\x00'
    vname_byte[0:len(vnamenull)] = bytearray(vnamenull, encoding='utf-8')

    head[0:VSIZE] =  vname_byte
    off = VSIZE*1
    
    # Store number of dimensions and recordsize to buffer
    ints = bytearray(8)
    head[off:off+8] = np.int32((len(dim),type2number(dtype)), order='c').tobytes()
    off += 8
    
    # Now write the actual dimensions as long (8 bytes)
    ints = bytearray(8*len(dim))
    ints[:] = np.int64(dim, order='c').tobytes()
    
    head[off::] = ints
    
    return head

# ==================================================================================
    
def unpack_header(head):
    vname = str(head[0:VSIZE].decode('utf-8','ignore').rstrip('\x00'))
    off = VSIZE*1
    tmp = np.frombuffer(head[off:off+8],dtype='int32')
    ndim = tmp[0]
    vtype = tmp[1]
    off+= 8
    dim = np.frombuffer(head[off:off+8*ndim], dtype='int64')
    return vname, vtype, dim

# ==================================================================================

def type2number(dtype):
    if(  dtype == 'uint8') : return 0
    elif(dtype == 'int8')  : return 1
    elif(dtype == 'uint16'): return 2
    elif(dtype == 'int16') : return 3    
    elif(dtype == 'uint32'): return 4
    elif(dtype == 'int32') : return 5
    elif(dtype == 'uint64'): return 6
    elif(dtype == 'int64') : return 7 
    elif(dtype == 'float32'):return 8
    elif(dtype == 'float64'):return 9
    elif(dtype == 'float128'):return 10
    else: return 12

# ==================================================================================

def number2dtype(num):
    if(  num == 0): return 'uint8'
    elif(num == 1): return 'int8'
    elif(num == 2): return 'uint16'
    elif(num == 3): return 'int16'
    elif(num == 4): return 'uint32'
    elif(num == 5): return 'int32'
    elif(num == 6): return 'uint64'
    elif(num == 7): return 'int64'
    elif(num == 8): return 'float32'
    elif(num == 9): return 'float64'
    elif(num == 10): return 'float128'
    else: return 'unknown'

# ==================================================================================

def number2size(num):
    return np.dtype(number2dtype(num)).itemsize

# ==================================================================================


class bio:
    def __init__(self, filename, mode='r', verbose=True):
        self.v = verbose
        self.m = mode
        self.filename = filename
        self.ini = False
        self.vsize = VSIZE
        
        # Check if file exists
        fexists = file_exists(self.filename)

        if(mode == 'r'):
            self.dat = open(self.filename, mode='rb')
            self._check_file_indicator()
            
        elif(mode == 'w'):
            self.dat = open(self.filename, mode='wb+')
            self._write_file_indicator()
        elif(mode == 'u'):
            if(not fexists):
                print('[error] bio::__init__: file does not exist -> '+filename)
                return
            
            self.dat = open(self.filename, mode='rb+')
            self._check_file_indicator()
        else:
            print('[error] bio::__init__: open mode not recognized, please use: r,w,u')
            self.dat = False
            return 

        
        
        if(self.v):
            print('[info] bio::__init__: opened file [{0}] in [{1}] mode'.format(self.filename, self.m))

        

    # ==================================================================================

    def __del__(self):
        if(self.v):
            print("[info] bio::__del__: closed file "+self.filename)
        self.dat.flush()
        self.dat.close()

    # ==================================================================================

    def _check_file_indicator(self):
        self.dat.seek(0)
        bu = bytearray(FSIZE)
        bu[:] = self.dat.read(FSIZE)

        ftype = str(bu[0:FSIZE-1].decode())
        if(ftype != 'IO01'):
            print('[error] bio::_check_file_indicator: incorrect file header for [{0}]'.format(self.filename))
            return

        
    # ==================================================================================

    def _write_file_indicator(self):
        bu = bytearray(FSIZE)
        bu[0:FSIZE-1] = bytearray("IO01", encoding='utf-8')

        self.dat.seek(0)
        self.dat.write(bu)
        self.dat.flush()


    # ==================================================================================

    def _read_head(self, vname):
        self.dat.seek(FSIZE)
        
    # ==================================================================================

    def variable_exists(self, vname):
        self.dat.seek(FSIZE)
        headsize = bytearray(4)

        exists = -1
        pos = self.dat.tell()
        while(1):
            ipos = pos*1
            headsize = np.frombuffer(self.dat.read(4), dtype='int32')
            if(not headsize): break
            head=bytearray(headsize)
            
            head[:] = self.dat.read(headsize[0])
            ivname, dtype, dim = unpack_header(head)
            if(vname == ivname):
                return ipos, vname, dtype, dim

            pos = self.dat.tell()
            
            pos += np.product(dim)*number2size(dtype)
            self.dat.seek(pos)
            
        return -1, None, None, None
    
    # ==================================================================================

    def create_empty_variable(self, vname, dim, dtype):
        pos, dum, dnum, idim  = self.variable_exists(vname)

        # Variable does not exist, go to the end of the file and create it there
        if(pos < 0):
            if(self.v):
                print('[info] bio::create_empty_variable: creating [{0}]'.format(vname))
            head = create_header(vname, dim, dtype)
            headsize = bytearray(4)
            headsize[:] = int.to_bytes(len(head),4,'little') #np.int32((len(head)), order='c').tobytes()
            
            self.dat.seek(0,2)
            self.dat.write(headsize)
            self.dat.write(head)
            self.dat.flush()
            
            pos = self.dat.tell() + np.product(np.int64(dim)) * np.dtype(dtype).itemsize - 1
            self.dat.seek(pos)
            self.dat.write(b'\0')
            self.dat.flush()
            return True
        else:
            inel = np.product(idim)
            isize = number2size(dnum)
            nel = np.product(dim)
            size = number2size(type2number(dtype))
            
            if((len(dim) == len(idim)) and (inel*isize == nel*size)):
                if(inel == nel): # variable exists and header and data are already in the right format
                    if(self.v):
                        print('[info] bio::create_empty_variable: variable [{0}] already exists in file [{1}] with correct dimensions and file type, not doing anything'.format(vname, self.filename))
                    return True

                else: # a variable exists, with the right dimensions but the header needs to be updated
                    head = create_header(vname, dim, dtype)
                    headsize = bytearray(4)
                    headsize[:] = int.to_bytes(len(head),4,'little') #np.int32((len(head)), order='c').tobytes()
                    
                    self.dat.seek(pos,0)
                    self.dat.write(headsize)
                    self.dat.write(head)
                    self.dat.flush()

                    return True
            else:
                print('[error] bio::create_empty_variable: there is already a variable named [{0}] in file [{1}] but the number of dimensions or the size of the datablock do not agree with the new variable. Unsupported case, returning.')
                return False
                    
                    
    # ==================================================================================

    def map_variable(self, vname):
        pos, dum, dnum, idim  = self.variable_exists(vname)

        if(pos < 0):
            print('[error] bio::map_variable: variable [{0}] does not exist in file [{1}]'.format(vname,self.filename))
            return None
        
        pos += get_header_size(len(idim))+np.dtype('int32').itemsize
        dtype = number2dtype(dnum)
        
        if(self.m == 'r'):
            return np.memmap(self.filename, shape=tuple(idim), dtype=dtype, offset=pos, mode='r')
        else:
            return np.memmap(self.filename, shape=tuple(idim), dtype=dtype, offset=pos, mode='r+')

    # ==================================================================================
    
    def read_variable(self, vname):
        
        pos, dum, dnum, idim  = self.variable_exists(vname)
        if(pos<0):
            print('[error] bio::read_variable: variable [{0}] does not exist in file [{1}]'.format(vname,self.filename))
            return None

        res = np.empty(tuple(idim), dtype=number2dtype(dnum), order='c')
        io = self.map_variable(vname)

        res[:] = io[:]
        return res
    
    # ==================================================================================

    def write_variable(self, vname, d):
        if(not self.create_empty_variable(vname, d.shape, d.dtype)):
            print('[error] bio::write_variable: could not create variable [{0}] in file [{1}]'.format(vname, self.filename))
            return False
        else:
            io = self.map_variable(vname)
            io[:] = d[:]
            return True
        
        
    # ==================================================================================

    def get_data_tree(self):
        res = []
        
        
        self.dat.seek(FSIZE)
        headsize = bytearray(4)

        exists = -1
        pos = self.dat.tell()
        while(1):
            ipos = pos*1
            headsize = np.frombuffer(self.dat.read(4), dtype='int32')
            if(not headsize): break
            head=bytearray(headsize)
            
            head[:] = self.dat.read(headsize[0])
            ivname, dtype, dim = unpack_header(head)

            a = container()
            a.vname = str(np.copy(ivname))
            a.dtype = number2dtype(dtype)
            a.dim = tuple(dim)
            res.append(a)
            
            pos = self.dat.tell()
            
            pos += np.product(dim)*number2size(dtype)
            self.dat.seek(pos)

        return res
    
    # ==================================================================================
