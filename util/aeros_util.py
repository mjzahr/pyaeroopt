import numpy as np

aeros_keys = ['STATIC','NONLINEAR','OUTPUT','NODES','TOPOLOGY','FORCE',
              'DISPLACEMENTS','ATTRIBUTE','MATERIAL']

def count_aeros_mesh_quantities(fname):

    f = open(fname,'r')
    # Count nodes, elements, dbc, fbc
    nnode = 0; in_node = False;
    nelem = 0; in_elem = False;
    ndbc  = 0; in_dbc  = False;
    nfbc  = 0; in_fbc  = False;
    nenxnel = 0;
    for line in f:
        sline = line.strip()
        if not sline: continue
        if sline[0] == '*': continue

        if sline in aeros_keys or 'Nodes' in sline or 'Elements' in sline:
            in_node = in_elem = in_dbc = in_fbc = False
            if sline == 'NODES' or 'Nodes ' in sline:
                in_node = True
            elif sline == 'TOPOLOGY' or 'Elements' in sline:
                in_elem = True
            elif sline == 'FORCE':
                in_fbc = True
            elif sline == 'DISPLACEMENTS':
                in_dbc = True
            continue
        if in_node: nnode+=1
        if in_dbc : ndbc +=1
        if in_fbc : nfbc +=1
        if in_elem:
            etype = int([s for s in sline.split(' ') if s][1])
            nelem+=1
            nenxnel+=(len([s for s in sline.split(' ') if s])-2)
    f.close()
    return nnode, nelem, ndbc, nfbc, nenxnel

def read_aeros_mesh(fname):

    # Count quantities pertaining to mesh
    nnode, nelem, ndbc, nfbc, nenxnel = count_aeros_mesh_quantities(fname)
    p = np.zeros((3,nnode), dtype=float, order='F')
    t_ptr = np.zeros(nelem+1, dtype=int, order='F')
    t = np.zeros(nenxnel, dtype=int, order='F')
    etype   = np.zeros(nelem,dtype=int,order='F')
    dbc_loc = np.zeros((3,nnode),dtype=bool,order='F')
    dbc_val = np.zeros(ndbc,dtype=float,order='F')
    fext    = np.zeros((3,nnode),dtype=float,order='F')

    nodecnt = elemcnt = dbccnt = fbccnt = -1

    f = open(fname)
    for line in f:
        sline = line.strip()
        if not sline: continue
        if sline[0] == '*': continue
        if sline in aeros_keys or 'Nodes' in sline or 'Elements' in sline:
            in_node = in_elem = in_dbc = in_fbc = False
            if sline == 'NODES' or 'Nodes ' in sline:
                in_node = True
            elif sline == 'TOPOLOGY' or 'Elements' in sline:
                in_elem = True
            elif sline == 'FORCE':
                in_fbc = True
            elif sline == 'DISPLACEMENTS':
                in_dbc = True
            continue

        split_line = [s for s in sline.split(' ') if s]
        if in_node:
            nodecnt+=1
            p[:,nodecnt] = [float(s) for s in split_line[1:]]
        if in_elem:
            elemcnt+=1
            tmp = int(split_line[1])
            if tmp == 23 or tmp == 5: # TET
                etype[elemcnt] = 0
            t_ptr[elemcnt+1] = t_ptr[elemcnt]+len(split_line[2:])
            t[t_ptr[elemcnt]:t_ptr[elemcnt+1]] = [int(s) for s in
                                                                 split_line[2:]]
        if in_dbc:
            dbccnt+=1
            nn = int(split_line[0])-1
            dof= int(split_line[1])-1
            dbc_loc[dof,nn] = True
            dbc_val[dbccnt] = float(split_line[2])

        if in_fbc:
            fbccnt+=1
            nn = int(split_line[0])-1
            dof= int(split_line[1])-1
            fext[dof,nn] = float(split_line[2])
    f.close()

    return p, t_ptr, t, etype, dbc_loc, dbc_val, fext
