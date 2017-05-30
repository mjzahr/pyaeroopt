import os, copy, subprocess
import numpy as np

from pyaeroopt.util.hpc_util import execute_code
from pyaeroopt.util.misc import count_lines, split_line_robust, is_numeric

def is_nodeset(fname):
    f = open(fname)
    line = f.readline().rstrip('\n')
    f.close()
    return line == 'FENODES'

def top_stats(fname, only_nodes=False):
    pass

def nodeset_stats(fname):
    return ['nodes', count_lines(fname)-2]

def xpost_stats(fname):

    nline = count_lines(fname)
    out = {'desc':None, 'nnode':None, 'ndof':None, 'nstep':None}

    f = open(fname,'r')
    for k, line in enumerate(f):
        if k == 0:
            out['desc']  = line.rstrip('\n')
        if k == 1:
            out['nnode'] = int(line.rstrip('\n'))
        if k == 3:
            out['ndof']  = len( [x for x in line.split(' ') if x] )
            break
    f.close()

    out['nstep'] = (nline-2)/(out['nnode']+1)
    return out

def read_top(fname, only_nodes=False):

    # Read all lines from file
    f = open(fname,'r')
    lines = f.readlines()
    f.close()

    nodes, elems = [], []
    for k, line in enumerate(lines):
        # Determine section currently reading from; create appropriate data
        # structures to store statistics and values of mesh.
        if 'Nodes' in line.lstrip(' ').lstrip('\t')[:5]:
            inNodes = True
            inElems = False

            ne = 0

            # Seek ahead to find next instance of Nodes or Elems to determine
            # size of node set
            for nn, lline in enumerate(lines[k+1:]):
                if ('Nodes' in lline.lstrip(' ').lstrip('\t')[:5] or
                    'Elements' in lline.lstrip(' ').lstrip('\t')[:8]):
                    break
            if lline is lines[-1]: nn+=1

            lineStrip = line.lstrip(' ').lstrip('\t')
            lineStrip = lineStrip.rstrip(' ').rstrip('\t').rstrip('\n')
            nodes.append([x for x in lineStrip.split(' ') if x])
            nodes[-1].append( np.zeros(nn,dtype=int) )
            nodes[-1].append( np.zeros((nn,3),dtype=float) )
            continue

        elif 'Elements' in line.lstrip(' ').lstrip('\t')[:8]:
            inNodes = False
            if only_nodes: continue

            inElems = True

            ne = 0

            # Use next line to determine number of nodes per element
            # ASSUMES ALL ELEMENTS IN ELEMENT SET ARE OF SAME TYPE
            nen = len( split_line_robust(lines[k+1]) ) - 2

            # Seek ahead to find next instance of Nodes or Elems to determine
            # size of element set
            for nn, lline in enumerate(lines[k+1:]):
                if ('Nodes' in lline.lstrip(' ').lstrip('\t')[:5] or
                    'Elements' in lline.lstrip(' ').lstrip('\t')[:8]):
                    break
            if lline is lines[-1]: nn+=1

            lineStrip = line.lstrip(' ').lstrip('\t')
            lineStrip = lineStrip.rstrip(' ').rstrip('\n').rstrip('\t')
            sep = '\t' if '\t' in lineStrip else ' '
            elems.append([x for x in lineStrip.split(sep) if x and
                                            x != 'using' and not is_numeric(x)])
            elems[-1].append( np.zeros(nn,dtype=int) )
            elems[-1].append( np.zeros(nn,dtype=int) )
            elems[-1].append( np.zeros((nn,nen),dtype=int) )

            continue

        # Split/strip line (lines might contain ' ' and \t as deliminaters)
        lineSplit = split_line_robust(line)

        # Add node number and coordinates to current node
        if inNodes:
            nodes[-1][-2][ne] = int(lineSplit[0])
            nodes[-1][-1][ne,:] = np.array([float(x) for x in lineSplit[1:]]
                                                               ).reshape(1,-1)

        # Add element number, type, and nodes to current element
        if inElems:
            elems[-1][-3][ne] = int(lineSplit[0])
            elems[-1][-2][ne] = int(lineSplit[1])
            elems[-1][-1][ne,:] = np.array([int(x) for x in lineSplit[2:]]
                                                               ).reshape(1,-1)
        ne+=1
    return ( nodes, elems )

def read_nodeset(fname):
    return np.loadtxt(fname, dtype='float', skiprows=1, comments='END')[:, 1:]

def read_xpost(fname):

    # Read all lines from file
    f = open(fname,'r')
    lines = f.readlines()
    f.close()

    nnode = int( lines[1].rstrip('\n') )
    ndof  = len( [x for x in lines[3].rstrip('\n').split(' ') if x] )
    nstep = ( len(lines) - 2 ) / ( nnode + 1)

    # Extract "time" information from lines
    time = np.array([float(x.rstrip('\n')) for x in lines[2:-1:nnode+1]])

    # Extract value information from lines
    val = np.zeros((nnode, ndof, nstep))
    for t, tim in enumerate(time):
        for k, line in enumerate(lines[3+t*(nnode+1):3+(t+1)*nnode+t]):
            val[k,:,t] = np.array( [float(x.rstrip('\n')) for
                                                    x in line.split(' ') if x] )
    return ( time, val )

def read_vmo(fname, coords=[0, 1, 2]):
    return np.loadtxt(fname, dtype='float', skiprows=3)[:, coords]

def read_der(fname, coords=[0, 1, 2]):
    return read_xpost(fname)[1][:, coords, :]

def write_top(fname, nodes, elems):

    f = open(fname,'w')

    # Write nodes
    for nodeset in nodes:
        f.write(nodeset[0]+' '+nodeset[1]+'\n')
        for k, nnum in enumerate(nodeset[2]):
            f.write(str(nnum)+'  '+str(nodeset[3][k,0])+' '+
                             str(nodeset[3][k,1])+' '+str(nodeset[3][k,2])+'\n')

    # Write elements
    for elemset in elems:
        f.write(elemset[0]+' '+elemset[1]+' using '+elemset[2]+'\n')
        for k, enum in enumerate(elemset[3]):
            f.write(str(enum)+'    '+str(elemset[4][k]))
            for e in elemset[5][k,:]:
                f.write('    '+str(e))
            f.write('\n')

    f.close()

def write_nodeset(nodes, fname, node_nums=None):
    f = open(fname,'w')
    f.write('FENODES\n')
    for k, node in enumerate(nodes):
        if node_nums is None:
            f.write(str(k+1)+' ')
        else:
            f.write(str(node_nums[k])+' ')
        f.write(str(node[0])+' '+str(node[1])+' '+str(node[2])+'\n')
    f.write('END')
    f.close()

def write_xpost(fname, name, tags, vals, header=None):

    # Shape of data to write
    if len(vals.shape) < 3: vals = vals[:, :, None]
    nnode, ndof, nstep = vals.shape
    
    # nstep nneds to match tags and vals
    if len(tags) != nstep:
        raise ValueError('len(tags) and vals.shape[2] must be equal')

    # Open file and write header
    f = open(fname,'w')
    f.write('Scalar ') if vals.shape[1]==1 else f.write('Vector ')
    f.write(header+' under load for '+name+'\n')
    f.write(str(nnode)+'\n')
    for t, time in enumerate(tags):
        f.write(str(time)+'\n')
        for v, val in enumerate(vals[:,:,t]):
            f.write('  '.join([str(vv) for vv in val])+'\n')
    f.close()

def combine_xpost(fname, files):

    # Statistics from xpost file
    stats = xpost_stats(files[0])
    nnode = stats['nnode']
    ndof  = stats['ndof']
    desc  = stats['desc']
    nfile = len(files)

    # Read/concatenate from each file
    time = np.zeros(nfile, dtype=float)
    val  = np.zeros((nnode, ndof, nfile), dtype=float)
    for k, file in enumerate(files):
        _, tmp = read_xpost(file)
        time[k]    = k
        val[:, :, k] = tmp[:, :, -1]

    # Write xpost file
    s = desc.split(' ')
    write_xpost(fname, s[-1], time, val, s[1])

def top2nodeset(top_name, nodeset_name):
    nodes, _ = read_top(top_name, True)
    write_nodeset(nodes[0][-1], nodeset_name, nodes[0][-2])

def flip_coord(fname, fname_flipped, swap=[0, 2, 1], ftype='top'):
    pass

def write_vmo(fname, dat, step_num=0, with_header=True, different_size=None):

    f = open(fname,'w')
    if with_header:
        f.write('Vector MODE under Attributes for nset\n')
        if different_size is not None:
            f.write(str(different_size)+'\n')
        else:
            f.write(str(dat.shape[0])+'\n')
    if step_num is not None:
        f.write('  {0:d}\n'.format(step_num))
    for d in dat:
        f.write(str(d[0])+'  '+str(d[1])+'  '+str(d[2])+'\n')
    f.close()

def write_der(fname, dat, vars=None):

    # Write all derivatives to file
    if vars is None and len(dat.shape) == 3:
        vars = np.arange(dat.shape[2])

    # Make sure var iterable
    if not hasattr(vars,'__getitem__'):
        vars = [vars]

    # If writing variable 0, open file and write header
    if vars[0] == 0:
        f = open(fname,'w')
        f.write('Vector MODE under Attributes for FluidNodes\n')
        f.write(str(dat.shape[0])+'\n')
    else:
        f = open(fname,'a')

    # Write derivatives
    for var in vars:
        f.write('  '+str(var)+'\n')
        for n in np.arange(dat.shape[0]):
            if len(dat.shape) == 2:
                f.write(str(dat[n,0])+'  '+str(dat[n,1])+'  '+
                                                            str(dat[n,2])+'\n')
            elif len(dat.shape) == 3:
                f.write(str(dat[n,0,var])+'  '+str(dat[n,1,var])+'  '+
                                                        str(dat[n,2,var])+'\n')
    f.close()


def read_multsoln(fname):
    multsoln = []
    f = open(fname, 'r')
    for line in f:
        multsoln.append(line.rstrip('\n'))
    f.close()
    return multsoln

def write_multsoln(fname, multsoln):
    f = open(fname, 'w')
    f.write('{0:d}'.format(len(multsoln)))
    for i, imultsoln in enumerate(multsoln):
        f.write('{0:s}'.format(imultsoln))
    f.close()

def flip_coord_top(fname_in, fname_out, swap=[0, 2, 1]):

    f_in  = open(fname_in, 'r')
    f_out = open(fname_out, 'w')
    for line in f_in:
        if 'Elements' in line:
            underNodes = False
        elif 'Nodes' in line:
            underNodes = True
            f_out.write(line)
            continue

        if underNodes:
            if swap == [0,1,2] or swap == [2,1,0] or swap ==  [1,0,2]: sign= 1.0
            else:                                                      sign=-1.0

            coords = [x.rstrip('\n') for x in line.split('\t') if x]
            f_out.write(coords[0]+' '+coords[swap[0]+1]+
                                  ' '+coords[swap[1]+1]+
                                  ' '+str(sign*float(coords[swap[2]+1]))+'\n')
        else:
            f_out.write(line)
    f_in.close()
    f_out.close()

def displace_top_with_vmo(fname, top, vmo):
    nodes, elems   = read_top(top)
    disp           = read_vmo(vmo)
    nodes[-1][-1] += disp
    write_top(fname, nodes, elems)

def displace_nodeset_with_vmo(fname, nodeset, vmo):
    nodes = read_nodeset(nodeset)+read_vmo(vmo)
    write_nodeset(fname, nodes)

def part_mesh(top, ndec, log=None, make_call=True, partmesh=None):

    # Default partnmesh executable
    if partmesh is None: partmesh = os.path.expandvars('$PARTMESH')

    # Build execution string, execute
    if make_call:
        top_old = copy.copy(top)
        top = top.split('/')[-1]+'.copy'
        subprocess.call('cp {0:s} {1:s}'.format(top_old, top), shell=True)
    exec_str = "{0:s} {1:s} {2:d}".format(partmesh, top, ndec)
    execute_code(exec_str, log, make_call)
    if make_call:
        subprocess.call('rm {0:s}'.format(top), shell=True)
        subprocess.call('mv {0:s}.dec.{2:d} {1:s}.dec.{2:d}'.format(top,
                                                     top_old, ndec), shell=True)
        top = copy.copy(top_old)
    return "{0:s}.dec.{1:d}".format(top, ndec) 

def sower_fluid_top(top, dec, cpus, nclust, geom_prefix, log=None,
                    make_call=True, sower=None):

    # Default sower executable
    if sower is None: sower = os.path.expandvars('$SOWER')

    # Build execution string, execute
    exec_str = "{0:s} -fluid -mesh {1:s} -dec {2:s}".format(sower, top, dec)
    for cpu in cpus:
        exec_str += " -cpu {0:d}".format(cpu)
    exec_str += " -cluster {0:d} -output {1:s}".format(nclust, geom_prefix)
    execute_code(exec_str, log, make_call)

def sower_fluid_extract_surf(msh, con, surf_top, bccode=-3, log=None,
                             make_call=True, sower=None):

    # Default sower executable
    if sower is None: sower = os.path.expandvars('$SOWER')

    # Build execution string, execute
    exec_str = ("{0:s} -fluid -merge -con {1:s} -mesh {2:s} "+
                "-skin {3:s} -bc {4:d}").format(sower, con, msh, surf_top,
                                                bccode)
    execute_code(exec_str, log, make_call)
    subprocess.call('mv {0:s}.xpost {0:s}'.format(surf_top), shell=True)

def sower_fluid_mesh_motion(mm_file, msh, con, out, bccode=-3, log=None,
                            make_call=True, sower=None):

    # Default sower executable
    if sower is None: sower = os.path.expandvars('$SOWER')

    # Build execution string, execute
    exec_str = ("{0:s} -fluid -split -con {1:s} -mesh {2:s} "+
                "-result {3:s} -ascii -bc {4:d} -out {5:s}").format(
                                          sower, con, msh, mm_file, bccode, out)
    execute_code(exec_str, log, make_call)

def sower_fluid_split(file2split, msh, con, out, from_ascii=True, log=None,
                      make_call=True, sower=None):

    # Default sower executable
    if sower is None: sower = os.path.expandvars('$SOWER')

    # Build execution string, execute
    exec_str = ("{0:s} -fluid -split -con {1:s} -mesh {2:s} "+
                "-result {3:s}").format(sower, con, msh, file2split)
    if from_ascii: exec_str = "{0:s} -ascii".format(exec_str)
    exec_str = "{0:s} -out {1:s}".format(exec_str, out)
    execute_code(exec_str, log, make_call)
    
def sower_fluid_merge(res_file, msh, con, out, name, from_bin=False, log=None,
                      make_call=True, sower=None):

    # Default sower executable
    if sower is None: sower = os.path.expandvars('$SOWER')

    # Build execution string, execute
    exec_str = "{0:s} -fluid -merge -con {1:s}".format(sower, con)
    exec_str = "{0:s} -mesh {1:s} -result {2:s}".format(exec_str, msh, res_file)
    exec_str = "{0:s} -name {1:s} -out {2:s}".format(exec_str, name, out)
    if from_bin: exec_str = "{0:s} -binary".format(exec_str)
    execute_code(exec_str, log, make_call)

def run_cd2tet_fromtop(top, out, log=None, make_call=True, cd2tet=None):

    # Default cd2tet executable
    if cd2tet is None: cd2tet = os.path.expandvars('$CD2TET')

    # Build execution string, execute
    exec_str = "{0:s} -mesh {1:s} -output {2:s}".format(cd2tet, top, out)
    execute_code(exec_str, log, make_call)

def run_xp2exo(top, exo_out, xpost_in=[], log=None, make_call=True,
               xp2exo=None):

    # Default xp2exo executable
    if xp2exo is None: xp2exo = os.path.expandvars('$XP2EXO')

    # Build execution string, execute
    exec_str = "{0:s} {1:s} {2:s}".format(xp2exo, top, exo_out)
    for res in xpost_in:
        exec_str = "{0:s} {1:s}".format(exec_str, res)
    execute_code(exec_str, log, make_call)

def meshtools_plane(res_file, msh, con, out, plane, log=None, make_call=True,
                    meshtools=None):
    pass

def split_nodeset(fname, npartition):
    X = read_nodeset(fname)
    stride = np.ceil( float(X.shape[0])/float(npartition) )
    for k in range(npartition):
        fname_partk = '{0:s}.{1:d}parts.part{2:s}'.format(fname, npartition,
                                             str(k).zfill(len(str(npartition))))
        if k == 0:
            write_nodeset(X[:stride, :], fname_partk)
        elif k == npartition-1:
            write_nodeset(X[k*stride:, :], fname_partk)
        else:
            write_nodeset(X[k*stride:(k+1)*stride, :], fname_partk)

def cat_split(fname_root, npartition, cleanup=False):
    s = ''
    for k in range(npartition):
        s += ' {0:s}.{1:d}parts.part{2:s}'.format(fname_root, npartition,
                                             str(k).zfill(len(str(npartition))))
    cat_str = 'cat {0:s} > {1:s}'.format(s, fname_root)
    execute_code(cat_str, None, True)
    if cleanup:
        rm_str  = 'rm {0:s}'.format(s)
        execute_code(rm_str, None, True)

def cat_split_gen(fname_root, fname_gen, n, cleanup=False):
    s = ''
    for k in range(n):
        s += ' {0:s}'.format(fname_gen(k))
    cat_str = 'cat {0:s} > {1:s}'.format(s, fname_root)
    execute_code(cat_str, None, True)
    if cleanup:
        rm_str  = 'rm {0:s}'.format(s)
        execute_code(rm_str, None, True)
