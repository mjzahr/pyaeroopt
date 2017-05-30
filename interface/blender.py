import sys, os, copy, subprocess
import numpy as np

from pyaeroopt.interface import CodeInterface
from pyaeroopt.util.misc import Group, Object
from pyaeroopt.util.blender_util import run_code
from pyaeroopt.util.frg_util import combine_xpost, run_xp2exo

class Blender(CodeInterface):
    """
    An object to facilitate interfacing to BLENDER.  Used to define the
    deformation of a point cloud from a list of modifiers (or compound
    modifiers)
    """
    def __init__(self, **kwargs):
        """
        Constructor
        """

        # Constructor of base class, check fields, and extract anticipated input
        super(Blender, self).__init__(**kwargs)

        if self.bin is None: self.bin = os.path.expandvars('$BLENDER')

        self.modifiers = Group([])
        self.dofs      = Group([])
        self.ndof      = 0
        self.method    = 'Additive'
        #self.method    = 'Sequential'
        self.deformee  = []
        self.eps       = 1.0e-4

        if self.method == 'Additive':
            print('Warning: Blender GUI will not show displacement output by '+
                  'blender_deform since it does sequential deformation.'+
                  'Additive deformation is a hack internal to pyAeroOpt. ' +
                  'To visualize deformation of surface/modifiers, create top '+
                  'files.')

    def execute(self, p, desc_ext=None, hpc=None, make_call=True):
        """
        Solve for displacements or derivatives using Blender from value of DOFs.

        Parameters
        ----------
        p : ndarray
          Vector of Design Variables or Abstract Variables
        """
        self.move_degrees_of_freedom(p)
        new_self = copy.deepcopy(self)

        self.create_input_file(p, desc_ext, self.db)
        run_code(p, new_self, self.ptcloud, self.vmo, self.der, self.eps,
                 self.xpost, self.log, make_call, self.bin, hpc.bg,
                 hpc.mpi, hpc.nproc) 

    def add_modifier(self, id, obj, weight=1.0):
        """
        Add a modifier to the blender object. The deformation will be the result
        of all modifiers acting on the object.

        Parameters
        ----------
        id : int
          Unique identifier
        obj : Modifier object
        """

        if obj.__class__.__name__ != 'Modifier':
            raise ValueError('obj must be an instance of the Modifier class')
        self.modifiers.add_to_group(Object(id, 'modifier', obj=obj,
                                           weight=weight, pos=self.modifiers.n))

    def add_degree_of_freedom(self, id, mod_ids, mod_dofs, expr=None):
        """
        Add degrees of freedom to Blender object.  Ability to combine DOFs of
        multiple nodes into single DOF (called Design Variable in SDESIGN
        terminology). Can also use an expression (python function) that will
        accept a vector of Abstract Variables (in SDESIGN terminology) and
        return the value of the Design Variable. If 'expr' specified for one
        Design Variable, it must be specified for all, and each must take
        arguments of the same length (nabsvar in declare_degrees_of_freedom).

        Parameters
        ----------
        id : int
        mod_ids : ndarray (or iterable) of int
          Modifier ids whose dofs (all or some) will be lumped into a Design
          Variable
        mod_dofs : list (len = len(mod_ids)) of lists of int
          Degree of Freedom for each modifier that will be lumped into the
          current Design Variable
        expr : function
          Map from the vector of (all) Abstract Variables to the present
          Design Variable (returns scalar)
        """

        # Ensure mod_ids, mod_dofs iterable
        if not hasattr(mod_ids ,'__iter__'):
            raise ValueError('mod_ids argument must be iterable')
        if type(mod_dofs) is not list:
            raise ValueError('mod_dofs argument must be a list of lists. '+
                             'If mod_dofs[i] only needs to be a scalar, it'+
                             'will be converted to list internally.')
        else:
            mod_dofs = [x if type(x) is list else [x] for x in mod_dofs]

        # Add degree of freedom
        self.dofs.add_to_group(Object(id, 'DOF', mod_ids=mod_ids,
                                      mod_dofs=mod_dofs, expr=expr))

    def make_degrees_of_freedom_from_modifiers(self):
        """
        Make a degree of freedom out of all of the degrees of freedom of each
        modifier (order is determined by the order of the modifiers added to
        object).
        """
        id = -1
        for mod in self.modifiers.get_from_id():
            for k in np.arange(mod.obj.ndof):
                self.add_degree_of_freedom(id, [mod.id], [k])

    def declare_degrees_of_freedom(self, vtype, nabsvar=0):
        """
        Function to declare degrees of freedom (type and number).  If Abstract
        Variables are used by specifying 'expr' in addDof, they must all accept
        the same number of arguments (nAbsVar).

        Parameters
        ----------
        vtype : str
          Type of degree of freedom
        nabsvar : int
          Number of abstract variables
        """

        self.vtype = vtype
        if vtype == 'dsgvar':
            self.ndof = 0
            for mod in self.modifiers.get_from_id():
                self.ndof += mod.obj.ndof
            #self.ndof = self.dofs.n
        elif vtype == 'absvar':
            self.ndof = nabsvar
            # Make sure 'expr' is not None for all degrees of freedom
            for item in self.dof.items:
                if item.expr is None:
                    raise ValueError('expr cannot be None for any Design '+
                                     'Variables if Abstract Variables declared')

    def convert_global_dofs_to_modifier_dofs(self, x):
        """
        Convert vector of global degrees of freedom into list of vector of
        degrees of freedom for each modifier.

        Parameters
        ----------
        x : ndarray
          Vector of Design Variables or Abstract Variables
        """

        # Create vector of DOFs for each modifier
        xmod = [np.zeros(mod.obj.ndof, dtype=float)
                                        for mod in self.modifiers.get_from_id()]

        for d, dof in enumerate(self.dofs.get_from_id()):

            # Extract value for degree of freedom
            if self.vtype == 'absvar':
                val = dof.expr(x)
            else:
                val = x[d]

            # Determine where to put value (which DOF of which modifier)
            for j, modId in enumerate(dof.mod_ids):
                mod = self.modifiers.get_from_id(modId)
                for k, mod_dof in enumerate(dof.mod_dofs[j]):
                    xmod[mod.pos][mod_dof]+=val 
        return xmod

    def move_degrees_of_freedom(self, x):
        """
        Move degrees of freedom of BLENDER object.

        Parameters
        ----------
        x : ndarray
          Vector of Design Variables or Abstract Variables
        """

        # Extract modifier DOFs from global DOFs
        xmod = self.convert_global_dofs_to_modifier_dofs(x)

        # For each Blender DOF, move corresponding DOFs for each modifier
        for mod in self.modifiers.get_from_id():
            mod.obj.move_degrees_of_freedom(xmod[mod.pos])

    def blender_make_deform_object_from_self(self):
        """
        Recursively make blender deform objects from the pyAeroOpt Deform
        objects in each of the Modifier objects. Only available during call
        to Blender.
        """
        for mod in self.modifiers.get_from_id():
            mod.obj.blender_make_deform_object_from_self()

    def blender_link_deform_objects(self):
        """
        Link all Deform objects of each Modifier.
        """
        # Loop over each Modifier and link
        for mod in self.modifiers.get_from_id():
            mod.obj.blender_link_deform_objects()

    def blender_add_deformee(self,ob):
        """
        Set the blender object to which the modifiers will be applied.  Multiple
        deformee can be added (i.e. modifiers will act on multiple nodesets).
        Only available during class to Blender.
        """
        self.deformee = ob
        for mod in self.modifiers.get_from_id():
            mod.obj.blender_add_deformee(ob)

    def blender_deform(self,x):
        """
        Apply deformation to deformeee from global degrees of freedom.

        Parameters
        ----------
        x : ndarray
          Global DOFs
        """

        from pyaeroopt.util.blender.mesh import extract_mesh
        from pyaeroopt.util.blender.mesh import extract_object_displacement

        # Extract modifier DOFs from global DOFs
        xmod = self.convert_global_dofs_to_modifier_dofs(x)

        # Different behavior for different deformation 'types'
        if self.method == 'Sequential':
            # For sequential, apply all modifiers to object
            for mod in self.modifiers.get_from_id():
                mod.obj.blender_deform(xmod[mod.pos])

            # Extract deformation
            disp = extract_object_displacement(self.deformee)

        elif self.method == 'Additive':
            # For additive, apply single modifier, extract displacement, and
            # reset modifier to zero so deformation is zero for application of
            # next modifier

            # First, set all deformation to zero and get displacement
            # (should be zero)
            for mod in self.modifiers.get_from_id():
                mod.obj.blender_deform(0.0*xmod[mod.pos])
            disp = np.zeros((len(extract_mesh(self.deformee).vertices), 3),
                            dtype=float)

            # Next, apply first modifier
            for k, mod in enumerate(self.modifiers.get_from_id()):
                if k > 0:
                    mod_prev = self.modifiers.get_from_id()[k-1]
                    mod_prev.obj.blender_deform(0.0*xmod[mod_prev.pos])

                mod.obj.blender_deform(xmod[mod.pos])
                disp += mod.weight*extract_object_displacement(self.deformee)
        return ( disp )

    def write_top(self, prefix, deformed=False, x=None, run_deform=True):

        # Extract modifier DOFs from global DOFs
        if deformed:
            xmod = self.convert_global_dofs_to_modifier_dofs(x)

        fnames = []
        for mod in self.modifiers.get_from_id():
            fname=["{0:s}{1:s}{2:s}.top".format(prefix,
                                                def_obj.__class__.__name__[0:4],
                                                str(def_obj.id).zfill(2))
                                          for def_obj in mod.obj.list_of_deform]
            fnames+=fname
            if deformed:
                mod.obj.write_top(fname, deformed, xmod[mod.pos],
                                       run_deform)
            else:
                mod.obj.write_top(fname)
        #subprocess.call('tar -zcf '+prefix+'.top.tgz '+
        #                                          '  '.join(fnames),shell=True)

    def write_xpost(self, prefix, x, run_deform=True):

        # Extract modifier DOFs from global DOFs
        xmod = self.convert_global_dofs_to_modifier_dofs(x)

        fnames = []
        for mod in self.modifiers.get_from_id():
            fname=["{0:s}{1:s}{2:s}.xpost".format(prefix,
                                                def_obj.__class__.__name__[0:4],
                                                str(def_obj.id).zfill(2))
                                          for def_obj in mod.obj.list_of_deform]
            fnames+=fname
            mod.obj.write_xpost(fname, xmod[mod.pos], run_deform)
        #subprocess.call('tar -zcf '+prefix+'.xpost.tgz '+
        #                                          '  '.join(fnames),shell=True)

    def combine_xpost(self,fname_prefix,prefixes):

        # Base filename for each deform object
        base=[]
        for mod in self.modifiers.get_from_id():
            base+=["{0:s}{1:s}".format(def_obj.__class__.__name__[0:4],
                                       str(def_obj.id).zfill(2))
                                          for def_obj in mod.obj.list_of_deform]

        # Get all filenames involved as list of lists
        all_fname = []
        for prefix in prefixes:
            fnames = ["{0:s}{1:s}.xpost".format(prefix, b) for b in base]
            all_fname.append(fnames)

        # Combine xpost files
        for k, b in enumerate(base):
            fnames_def_i = [f[k] for f in all_fname]
            combine_xpost("{0:s}{1:s}.xpost".format(fname_prefix, b),
                          fnames_def_i)

    def write_exo(self, prefix_top, prefix_xpost, prefix_exo, log=None,
                  make_call=True, xp2exo=None):

        # Base filename for each deform object
        base=[]
        for mod in self.modifiers.get_from_id():
            base+=["{0:s}{1:s}".format(def_obj.__class__.__name__[0:4],
                                       str(def_obj.id).zfill(2))
                                          for def_obj in mod.obj.list_of_deform]

        # Run xp2exo
        for k, b in enumerate(base):
            run_xp2exo("{0:s}{1:s}.top".format(prefix_top, b),
                       "{0:s}{1:s}.exo".format(prefix_exo, b),
                       ["{0:s}{1:s}.xpost".format(prefix_xpost, b)],
                       log, make_call, xp2exo)
