from glob import glob
import numpy as np

from pyaeroopt.util.misc import count_lines

def read_ascii_output(fname, ftype):
    """
    Read Aero-F ascii outputs, either unsteady or sensitivity type files.

    Input
    -----
      fname : str
        filename of output file
      ftype : str
        type of file (unsteady or sensitivity)

    Output
    ------
      data - ndarray
        (if ftype != sensitivity) returns all information from an unsteady
        ascii output file, including timesteps, in a nstep x n matrix
      act  - ndarray
        (if ftype == sensitivity) returns active variable for each derivative
      val  - ndarray
        (if ftype == sensitivity) returns value of output(s)
      der  - ndarray
        (if ftype == sensitivity) returns derivatives of output(s)
    """

    # Check if file exists; if not return inf
    exists = glob(fname)
    if exists: nLines = count_lines(fname)
    if not exists or nLines == 1:
        return(None)

    data = np.loadtxt(fname, skiprows=1)
    if 'sensitivity' not in ftype.lower():
        return ( data )
    else:
        if len( data.shape ) > 1:
            nout = (data.shape[1]-2)/2.0
            act  = data[:,1]
            val  = data[-1,2:2+nout]
            der  = data[:,2+nout:]
        else:
            nout = 1
            act  = data[1]
            val  = data[2:2+nout]
            der  = data[2+nout:]

        return( act, val, der)

def extract_ascii_output(dat, which='Lx', typ='steady'):
    """
    Compute output from ascii file.  Possible options are: Lx, Ly, Lz, Fx, Fy,
    Fz, Mx, My, Mz, MP (match pressure), FN (flux norm).

    Input
    -----
    dat - tuple of ndarray
      Output of read_ascii_output
    which - str
      Output to extract from ascii file
    typ   - str
      Output corresponds to 'steady' or 'unsteady' output format

    Output
    ------
    out - ndarray, or tuple of ndarray
      Output of interest (and sensitivity, if applicible)
    """

    col = ascii_column_of_output(which)

    # Extract appropriate columns
    if dat is None: return None
    if typ == 'steady':
        # First output is the value, the second will be the derivative,
        # and the third will be the active variable (variable to which value
        # is being differentiated).
        if len(dat[2].shape) == 1:
            return (dat[1][col], dat[2][col], dat[0])
        else:
            return (dat[1][col], dat[2][:,col], dat[0])
    else:
        # First output will be time stepping data, second will be values
        if len(dat.shape) == 1: dat = dat[None, :]
        return (dat[:,:4], dat[:,4+col])

def read_extract_ascii_output(fname, ftype, which='Lx'):
    """
    Read Aero-F ascii outputs and return output of interest.
    See read_ascii_output, extract_ascii_output for inputs/outputs.
    """

    dat = read_ascii_output(fname, ftype)
    typ = 'steady' if 'sensitivity' in ftype.lower() else 'unsteady'
    return extract_ascii_output(dat, which, typ)

def ascii_column_of_output(which):
    """
    Determine column of output in AERO-F ascii file.

    Input
    -----
    which - str
      Output to extract from ascii file

    Output
    ------
    out - int
      Column number of output of interest (0-based)
    """

    # Lift/Drag
    if which == 'Lx': col = 0
    if which == 'Ly': col = 1
    if which == 'Lz': col = 2
    # Forces/moments
    if which == 'Fx': col = 0
    if which == 'Fy': col = 1
    if which == 'Fz': col = 2
    if which == 'Mx': col = 3
    if which == 'My': col = 4
    if which == 'Mz': col = 5
    # Match pressure
    if which == 'MP': col = 0
    # Flux norm
    if which == 'FN': col = 0

    return col

def write_ascii_input(fname, entries=[]):
    """
    Write AERO-F database input file.

    Input
    -----
    fname - str
      Filename of file to be written
    entries - list or list of list
      MultipleSolution: ['sol1', 'sol2', ...]
      Snapshots       : [['snap1', start1, finish1, freq1, weight1],
                         ['snap2', start2, finish2, freq2, weight2], ...]

    Output
    ------
    None, file is written
    """
    if entries is None: entries=[]

    # Make entries unique
    if len(entries) == 0:
        unique_entries = []
    else:
        if type(entries[0]) == list:
            unique_entries = [list(xx) for xx in set(tuple(x) for x in entries)]
        else:
            entries       = [s.rstrip('\n') for s in entries]
            unique_entries = list(set(entries))

    # Write unique entries to file
    f = open(fname, 'w')
    if len(unique_entries) == 0:
        f.write('0\n')
    elif len(unique_entries) > 0:
        # First line is number of entries
        f.write(str(len(unique_entries))+'\n')
        for entry in unique_entries:
            # Determine if writing single string or iterable to line
            if isinstance(entry, str):
                f.write(entry+'\n')
            elif isinstance(entry, (list, tuple)):
                for x in entry:
                    f.write(str(x)+'   ')
                f.write('\n')
    f.close()

def write_ascii_multsoln_param(fname, multsoln, param):
    """
    Write AERO-F database (multiple solutions ans parameters) input file.

    Input
    -----
    fname - str
      Filename of file to be written
    multsoln - list of str
      Filename of solutions files
    param - list of array
      Parameters corresponding to each solution file, require
      len(multsoln) == len(param)

    Output
    ------
    None, file is written
    """

    # Check compatibility of multsoln and param
    if len(multsoln) != len(param):
        raise ValueError('multsoln and param must be same size')

    # Write entries to file
    f = open(fname, 'w')
    if len(multsoln) == 0:
        f.write('0\n0\n')
    else:
        # First line is number of entries, second line is number of parameters
        f.write(str(len(multsoln))+'\n')
        f.write(str(len(param[0]))+'\n')
        for j, jmultsoln in enumerate(multsoln):
            f.write(jmultsoln+'\n')
            for pk in param[j]:
                f.write(str(pk)+'\n')
    f.close()
