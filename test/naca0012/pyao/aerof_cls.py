#TODO: Use db (database) to CREATE filename!
import numpy as np

from pyaeroopt.interface.aerof import Aerof, AerofInputFile
from pyaeroopt.test.naca0012.pyao.aerof_blk import *

class AerofHdm(Aerof):
    """
    p : numpy array
        Shape parameters
    desc : list
        [type, reconst, mach, alpha, beta]
    desc_ext : list
        [multsoln]
    """
    def __init__(self, **kwargs):
        super(AerofHdm, self).__init__(**kwargs)

    def set_mach(self, mach):
        self.desc[2]=mach
        return self

    def set_alpha(self, alpha):
        self.desc[3]=alpha
        return self

    def set_beta(self, beta):
        self.desc[4]=beta
        return self

    def create_input_file(self, p, desc_ext, db=None):

        prob.Type                        = self.desc[0]
        spac.NavierStokes.Reconstruction = self.desc[1]
        bounCond.Inlet.Mach              = self.desc[2]
        bounCond.Inlet.Alpha             = self.desc[3]
        bounCond.Inlet.Beta              = self.desc[4]
        if desc_ext[0] is not None: prob.MultipleSolutions = desc_ext[0]

        prefix = 'aerof'
        def append_prefix(s): return "{0:s}.{1:s}".format(prefix, s)
        outp.Postpro.LiftandDrag = append_prefix('liftdrag')
        outp.Postpro.Pressure    = append_prefix('pressure')
        outp.Restart.Solution    = append_prefix('sol')
        outp.Restart.Restart     = append_prefix('rst')

        mach   = self.desc[2]
        time.CflMax = np.min([1.0e4,np.max([100.0,100.0e1*(mach-0.3)/0.4 +
                                                        1.0e4*(0.7-mach)/0.4])])

        fname = append_prefix('input')
        log   = append_prefix('log')
        self.infile = AerofInputFile(fname, [prob, inpu, outp, surf, equa,
                                             bounCond, spac, time], log)

class AerofRom(Aerof):
    def __init__(**kwargs):
        super(AerofRom, self).__init__(**kwargs)

    def create_input_file(self, p, desc_ext, db=None):
        # Use p, desc_ext, self.desc to create input file
        self.infile = None

class AerofGnat(Aerof):
    def __init__(**kwargs):
        super(AerofGnat, self).__init__(**kwargs)

    def create_input_file(self, p, desc_ext, db=None):
        # Use p, desc_ext, self.desc to create input file
        self.infile = None
