import os

from pyaeroopt.util.hpc_util import execute_str, mpi_execute_str

machine_db = {'independence':{'ppn':12, 'scheduler':'maui',
                              'maxwall':[99.0*24.0], 'queue':['standard']},
              'kratos'      :{'ppn':16, 'scheduler':'slurm', 'maxwall':[48.0],
                              'queue':['standard']},
              'copper'      :{'ppn':32, 'scheduler':'maui',
                              'maxwall':[120.0,1.0,4.0],
                              'queue':['standard', 'debug', 'background']}}

class Hpc(object):
    """
    High-Performance Computing object

    Data Members
    ------------
    mpi : str
        Path to mpiexec
    machinefile: str
        Path to machinefile to be passed to mpiexec
    machine: str
        Name of machine in use, must be contained machine_db if batch is True
    batch: bool
        Indicates whether to run job in batch mode, implies writing of PBS
    """
    def __init__(self, **kwargs):

        self.mpi = os.path.expandvars('$MPI')
        self.machinefile = None
        if 'mpi' in kwargs: self.mpi = kwargs.pop('mpi')
        if 'machinefile' in kwargs: self.machinefile = kwargs.pop('machinefile')
        self.machine = kwargs.pop('machine') if 'machine' in kwargs else None
        self.batch   = kwargs.pop('batch')   if 'batch'   in kwargs else False
        self.bg      = kwargs.pop('bg')      if 'bg'      in kwargs else False
        self.nproc   = kwargs.pop('nproc')   if 'nproc'   in kwargs else 1
        self.nompi   = kwargs.pop('nompi')   if 'nompi'   in kwargs else False

    @property
    def ppn(self):
        ppn = None
        if self.machine is not None: ppn = machine_db[self.machine]['ppn']
        return ppn

    def execute_str(self, bin, infile):
        if self.nompi:
            return execute_str(bin, infile, self.bg)
        else:
            return mpi_execute_str(bin, infile, self.nproc, self.mpi, self.bg,
                                   self.machinefile)
