import os

from pyaeroopt.interface.interface import CodeInterface
from pyaeroopt.util.frg_util import part_mesh, sower_fluid_top
from pyaeroopt.util.frg_util import sower_fluid_extract_surf
from pyaeroopt.util.frg_util import sower_fluid_mesh_motion
from pyaeroopt.util.frg_util import sower_fluid_split
from pyaeroopt.util.frg_util import sower_fluid_merge
from pyaeroopt.util.frg_util import run_xp2exo, meshtools_plane

class Frg(CodeInterface):
    """
    An object to facilitate interfacing to FRG utility codes: SOWER, PARTNMESH,
    MESHTOOLS, XP2EXO
    """
    def __init__(self, **kwargs):

        super(Frg, self).__init__(**kwargs)

        # Default executable locations
        self.sower     = os.path.expandvars('$SOWER')
        self.partmesh  = os.path.expandvars('$PARTMESH')
        self.meshtools = os.path.expandvars('$MESHTOOLS')
        self.xp2exo    = os.path.expandvars('$XP2EXO')

        # Extract anticipated inputs
        if 'sower'     in kwargs: self.sower     = kwargs.pop('sower')
        if 'partmesh'  in kwargs: self.partmesh  = kwargs.pop('partmesh')
        if 'meshtools' in kwargs: self.meshtools = kwargs.pop('meshtools')
        if 'xp2exo'    in kwargs: self.xp2exo    = kwargs.pop('xp2exo')

        self.geom_pre = kwargs.pop('geom_pre') if 'geom_pre' in kwargs else None
        self.top      = kwargs.pop('top')      if 'top'      in kwargs else None
        self.surf_top = kwargs.pop('surf_top') if 'surf_top' in kwargs else None
        self.surf_nodeset = None
        if 'surf_nodeset' in kwargs:
            self.surf_nodeset = kwargs.pop('surf_nodeset')

        self.msh = self.geom_pre+'.msh' if self.geom_pre is not None else None
        self.con = self.geom_pre+'.con' if self.geom_pre is not None else None

    def part_mesh(self, ndec, log='partmesh.log', make_call=True):
        self.dec = part_mesh(self.top, ndec, log, make_call, self.partmesh)

    def sower_fluid_top(self, cpus, nclust, log='sower.top.log',
                        make_call=True):
        sower_fluid_top(self.top, self.dec, cpus, nclust, self.geom_pre,
                        log, make_call, self.sower)

    def sower_fluid_extract_surf(self, bccode=-3, log='sower.extract.log',
                                 make_call=True):
        sower_fluid_extract_surf(self.msh, self.con, self.surf_top, bccode, log,
                                 make_call, self.sower)

    def sower_fluid_mesh_motion(self, mm_file, out, bccode=-3,
                                log='sower.mm.log', make_call=True):
        sower_fluid_mesh_motion(mm_file, self.msh, self.con, out, bccode, log,
                                make_call, self.sower)

    def sower_fluid_split(self, file2split, out, from_ascii=True,
                          log='sower.split.log', make_call=True):
        sower_fluid_split(file2split, self.msh, self.con, out, from_ascii, log,
                          make_call, self.sower)

    def sower_fluid_merge(self, res_file, out, name, from_bin=False,
                          log='sower.merge.log', make_call=True):
        sower_fluid_merge(res_file, self.msh, self.con, out, name, from_bin,
                          log, make_call, self.sower)

    def run_xp2exo(self, exo_out, xpost_in, log='xp2exo.log', make_call=True):
        run_xp2exo(self.top, exo_out, xpost_in, log, make_call, self.xp2exo)

    def meshtools_plane(self, res_file, out, plane, log='meshtools.plane.log',
                        make_call=True):
        meshtools_plane(res_file, self.msh, self.con, out, plane, log,
                        make_call, self.meshtools)
