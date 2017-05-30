import os, subprocess, copy
import numpy as np

from pyaeroopt.util.misc import Object, Group
from pyaeroopt.util.frg_util import write_top, write_xpost
from pyaeroopt.util.hpc_util import execute_str, execute_code

try:
    import bpy
    from mathutils   import Vector, Euler, Matrix
    from pyaeroopt.util.blender.mesh     import create_mesh_object
    from pyaeroopt.util.blender.mesh     import extract_object_displacement
    from pyaeroopt.util.blender.modifier import add_modifier
    from pyaeroopt.util.blender.modifier import create_lattice
    from pyaeroopt.util.blender.modifier import create_mesh_deform, bind_cage
    from pyaeroopt.util.blender.modifier import create_armature
    from pyaeroopt.util.blender.modifier import create_bones_in_armature
except ImportError:
    pass

class Modifier:
    """
    Modifier class that will be used to deform the nodes of a point clound.  The
    modifier is defined by a list of Deform objects (Lattice, Skeleton, Cage,
    etc) where the first object in this list defines DOFs of the modifier and
    the last object defines the deformation of a point cloud; the ith object
    in this list will be used to deform the nodes of the i+1 object; the DOFs
    of the intermediate object will be ignored and the nodes of the object
    will be used to define the deformation.
    """
    def __init__(self, id, list_of_deform):
        """
        Constructor for Modifier class.

        Parameters
        ----------
        id : int
          Unique identifier of Modifier object
        list_of_deform : list of Deform objects
          Defines the modifier; list_of_deform[0] defines the degrees of freedom
          of the modifier (modifier inherits DOFs from list_of_deform[0]) and
          list_of_deform[-1] defines the actual deformation object to be applied
          to a nodeset.  The behavior will be as follows: list_of_deform[i] will
          be used to deform the nodes of list_of_deform[i+1] until
          list_of_deform[-1] which will deform the nodes in some nodeset of
          interest.
        """

        self.id           = id
        self.list_of_deform = ( list_of_deform if type(list_of_deform) is list 
                                                         else [list_of_deform] )
        self.dofs         = list_of_deform[0].dofs
        self.ndof         = list_of_deform[0].ndof

    def move_degrees_of_freedom(self,x):
        """
        Move degrees of freedom of Modifier object.

        Parameters
        ----------
        x : list or ndarray of float
          value of Design Variables or Abstract Variables for list_of_deform[0]
          Deform object (Modifier inherits DOFs from list_of_deform[0]).
        """
        self.list_of_deform[0].move_degrees_of_freedom(x)

    def blender_make_mesh_object_from_nodes(self):
        """
        Make blender mesh object from nodes of the pyaeroopt Deform objects in
        list_of_deform.  Only available during call to Blender.
        """
        for def_obj in self.list_of_deform:
            def_obj.blender_make_mesh_object_from_nodes()

    def blender_make_deform_object_from_self(self):
        """
        Make blender deform objects from the pyAeroOpt Deform objects in
        list_of_deform.  Only available during call to Blender.
        """
        for def_obj in self.list_of_deform:
            def_obj.blender_make_deform_object_from_self()

    def blender_link_deform_objects(self):
        """
        Link blender objects such that the nodes of list_of_deform[i] is
        modified by list_of_deform[i-1].
        """
        # Loop over Deform objects and link appropriately
        for k, def_obj in enumerate(self.list_of_deform):
            if k == len(self.list_of_deform)-1: continue
            def_obj.blender_add_deformee(
                                 self.list_of_deform[k+1].blend_objs.deform_obj)
            try:
                def_obj.blender_add_deformee(
                                   self.list_of_deform[k+1].blend_objs.mesh_obj)
            except:
                continue

    def blender_add_deformee(self, ob):
        """
        Set the blender object to which the (sequence of) modifiers will be
        applied.
        """
        self.list_of_deform[-1].blender_add_deformee(ob)

    def blender_deform(self, x):
        """
        Apply deformation to 'nodeset' by using list_of_deform[i] to deform 
        list_of_deform[i+1] as described in comments of constructor (__init__).
        Requires call to blender.  Option to write top file and xpost file for
        every level of deformation can be specified using **kwargs.

        Parameters
        ----------
        x : ndarray
          DOFs of Deform object in list_of_deform[0]

        Warning
        -------
        Assumes blender objects created and linked.
        """
        # Loop over Deform objects in list
        def_obj = self.list_of_deform[0]
        def_obj.blender_deform(x, 'dof')

    def write_top(self, fname, deformed=False, x=None, run_deform=True):
        """
        Write top file for all Deform objects of Modifier. Option to write
        undeformed or deformed nodes.

        Parameters
        ----------
        fname : str
          Filename of top file
        deformed : bool
          Defines whether to add disp to nodes prior to writing nodes
        """

        # Write top file for each Deform object
        if not deformed:
            for k, def_obj in enumerate(self.list_of_deform):
                def_obj.write_top(fname[k])
        else:
            self.list_of_deform[0].write_top(fname[0],deformed,x,
                                                   self.list_of_deform[0].vtype)

            # Blender modules
            if run_deform:
                self.blender_deform(x)
            for k, def_obj in enumerate(self.list_of_deform[1:]):
                disp = def_obj.blender_extract_deformation()
                def_obj.write_top(fname[k+1],deformed,disp,'nodes')

    def write_xpost(self,fname,x,run_deform=True):
        """
        Write xpost file for all Deform objects of Modifier.

        Parameters
        ----------
        fname : str
          Filename of xpost file
        """

        # Write xpost file for each Deform object
        self.list_of_deform[0].write_xpost(fname[0],x,
                                                   self.list_of_deform[0].vtype)

        # Blender modules
        if run_deform:
            self.blender_deform(x)
        for k, def_obj in enumerate(self.list_of_deform[1:]):
            disp = def_obj.blender_extract_deformation()
            def_obj.write_xpost(fname[k+1],disp,'nodes')

class Deform:
    """
    Deform class that will be the superclass for various blender modifier
    objects (Lattice, Cage, Skeleton, etc). Characterized by an id and nodes;
    individual subclasses have additional properties.  Ability to add/move
    degrees of freedom that can lump node motion into design variables
    (expressions can be used to map from some abstract variable to the
    motion of the design variable that will in turn map to the motion of the
    object nodes).
    """

    def __init__(self,id=0):
        """
        Deform object constructor.

        Parameters
        ----------
        id : int
          Unique identifier of Deform object
        nodes : ndarray
          nnode x 3 array defining nodes of Deform object
        disp : ndarray
          nnode x 3 array defining nodal displacements of Deform object
        """

        # Defaults/inputs
        self.id    = id
        self.vtype  = 'nodes'
        self.nodes = []
        self.dofs  = Group([])
        self.ndof  = 0

        # Object to hold all blender-specific data structures
        self.blend_objs = Object(0, 'blender-objects', deform_obj=[],
                                 deform_type='', deformee_obj=[])

    def add_degree_of_freedom(self, id, nodes, dirs, expr=[]):
         """
         Add degrees of freedom to Deform object.  Ability to combine DOFs of
         multiple nodes into single DOF (called Design Variable in SDESIGN
         terminology). Can also use an expression (python function) that will
         accept a vector of Abstract Variables (in SDESIGN terminology) and
         return the value of the Design Variable. If 'expr' specified for one
         Design Variable, it must be specified for all, and each must take
         arguments of the same length (nabsvar in declare_degrees_of_freedom).

         Parameters
         ----------
         id : int
         nodes : ndarray (or iterable)
           Node number of nodes whose dofs (all or some; specified in dir) will
           be lumped into a Design Variable
         dirs : list (len = nodes.size) of lists (len = 1, 2, 3) of int
           Degrees of freedom for each node that will be lumped into the current
           Design Variable
         expr : function
           Map from the vector of (all) Abstract Variables to the present
           Design Variable (returns scalar)
         """
 
         # Ensure nodes iterable and dirs list of lists
         if not hasattr(nodes,'__iter__'):
             raise ValueError('nodes argument must be iterable')
         if type(dirs) is not list:
             raise ValueError('dirs argument must be list of lists. If '+
                              'dirs[i] only needs to be a scalar, it will be'+
                              'converted to list internally.')
         else:
             dirs = [x if type(x) is list else [x] for x in dirs]
 
         # Add degree of freedom
         self.dofs.add_to_group( Object(id, 'DOF', nodes=nodes,
                                      dirs=dirs, expr=expr) )

    def make_degrees_of_freedom_from_nodes(self, dirs):
        """
        Loops over all nodes and adds between 1 and 3 DOFs per node (depending
        on which directions to convert to DOFs as specified in 'dirs'). Helper
        function to avoid doing this task manually for the common case of
        using all nodes as DOFs.

        Parameters
        ----------
        dirs : int or iterable of ints
          Directions to turn into degree of freedom at each node
        """

        # Ensure dirs is int or iterable of ints
        if not ( (type(dirs) is int) or
                          (hasattr(dirs,'__iter__') and type(dirs[0]) is int) ):
            raise ValueError('dirs must be int or iterable of ints')

        id = -1
        for k, node in enumerate(self.nodes):
            for dir in dirs:
                id += 1
                self.add_degree_of_freedom(id,[k],[dir])

    def make_degrees_of_freedom_from_elems(self, dirs):
        """
        Loops over all nodes and adds between 1 and 3 DOFs per element
        (depending on which directions to convert to DOFs as specified in
        'dirs'). Helper function to avoid doing this task manually for the
        common case of using all nodes as DOFs. These DOFs correspond to
        rotations in the x, y, z planes.

        Parameters
        ----------
        dirs : int or iterable of ints
          Directions to turn into degree of freedom at each node
        """

        # Ensure dirs is int or iterable of ints
        if not ( (type(dirs) is int) or
                          (hasattr(dirs,'__iter__') and type(dirs[0]) is int) ):
            raise ValueError('dirs must be int or iterable of ints')

        id = -1
        for k, edge in enumerate(self.edges):
            for dir in dirs:
                id += 1
                self.add_degree_of_freedom(id,[k],[dir])

    def declare_degrees_of_freedom(self, vtype, nabsvar=0):
        """
        Function to declare degrees of freedom (type and number).  If Abstract
        Variables are used by specifying 'expr' in addDof, they must all accept
        the same number of arguments (nabsvar).

        Parameters
        ----------
        vtype : str
          Type of degree of freedom ('absvar', 'dsgvar')
        nabsvar : int
          Number of Abstract Variables
        """

        self.vtype = vtype
        if vtype == 'nodes':
            self.ndof = self.nodes.size
        elif vtype == 'dsgvar':
            self.ndof = self.dofs.n
        elif vtype == 'absvar':
            self.ndof = nabsvar
            # Make sure 'expr' is not None for all degrees of freedom
            for item in self.dofs.items:
                if len(item.expr) == 0:
                    raise ValueError('expr cannot be None for any Design '+
                                     'Variables if Abstract Variables declared')

    def move_degrees_of_freedom(self, x):
        """
        Move degrees of freedom of Deform object.

        Parameters
        ----------
        x : float
          Value of Design Variable or Abstract Variables used to deform object
        """

        # Matrix to hold displacments of nodes
        disp = np.zeros(self.nodes.shape,dtype=float)

        for d, dof in enumerate(self.dofs.get_from_id()):
            # Extract value for degree of freedom
            if self.vtype == 'absvar':
                val = 0.0
                for expr in dof.expr:
                    val += expr(x, dof, self.nodes)
            else:
                val = x[d]

            # Apply motion to appropriate nodes/directions
            for j, nnum in enumerate(dof.nodes):
                for k, dir in enumerate(dof.dirs[j]): 
                    disp[ nnum , dir ] = val

        return(disp)

    def deform(self, x, vtype, **kwargs):
        """
        Deform object by moving the abstract variables, design variables, or
        nodes of the deformation object. Option to write top file and xpost
        (disp) file specified through kwargs.

        Parameters
        ----------
        x - ndarray
          Vector containing value of absvar, dsgvar, or nodes
        vtype - str
          Type of degree of freedom ('absvar', 'dsgvar', 'nodes')
    
        Output
        ------
        disp : ndarray
          Displacements of nodes of Deform object
        """

        # Compute disp from x
        if vtype == 'nodes':
            disp = x
        else:
            disp = self.move_degrees_of_freedom(x)

        # Return displacement of nodes of object
        return ( disp )

    def blender_make_deform_object_from_self(self):
        """
        Make blender object from Deform object. Implemented in derived classes.
        """
        print("Implement me for your subclass of DEFORM!")

    def blender_apply_deform_object_to_mesh_object(self, ob, **kwargs):
        """
        Apply blender modifier to blender object. Only available during call to
        Blender.
        """
        mod=add_modifier(ob, self.blend_objs.deform_obj,
                        self.blend_objs.deform_type)
        return mod

    def blender_add_deformee(self, ob):
        """
        Set the blender object to which the (sequence of) modifiers will be
        applied. Only availabe during call to Blender.
        """
        self.blender_apply_deform_object_to_mesh_object(ob)
        self.blend_objs.deformee_obj.append(ob)

    def blender_deform(self, x, vtype, **kwargs):
        """
        Apply deformation to deformee from values of node displacement, dsgvar,
        or absvar.  First, pass 'x' to deform pyAeroOpt Deform object, then
        impose this deformation on the blender modifiers.

        Parameters
        ----------
        x : ndarray
          Array containing nodal displacement, dsgvar, or absvar
        vtype : str
          Type of degree of freedom ('dof' for 'absvar' or 'dsgvar', 'nodes')
        """
        print("Implement me for your subclass of DEFORM!")


    def get_nodes(self, deformed=False, x=None, vtype=None):
        """
        Return (deformed) nodes of Deform object.
        """

        nodes = copy.copy( self.nodes )

        # Compute displacement if deformed lattice requested
        if deformed:
            print('in deformed')
            disp  =  self.deform(x, vtype)
            print(np.linalg.norm(disp))
            nodes += disp

        return ( nodes )

    def get_connectivity(self):
        """
        Get connectivity of mesh
        """
        pass

class Lattice(Deform):
    """
    Lattice Deform class.
    """
    def __init__(self, id, part=(2,2,2), center=(0.0,0.0,0.0),
                 scale=(1.0,1.0,1.0), rot=(0.0,0.0,0.0), **kwargs):
        """
        Constructor
        """

        # Inputs/defaults
        Deform.__init__(self,id)

        self.part   = part
        self.center = center
        self.scale  = scale
        self.rot    = rot
        if rot is None:
            rot = {'vec': np.array([1.0, 0.0, 0.0]), 'angle': 0.0}

        self.weight  = kwargs['weight'] if 'weight' in kwargs else 1.0
        self.interp  = kwargs['interp'] if 'interp' in kwargs else 'BSPLINE'

        # Lattice points
        Lx = self.scale[0]; dx = Lx / (float(self.part[0]-1))
        Ly = self.scale[1]; dy = Ly / (float(self.part[1]-1))
        Lz = self.scale[2]; dz = Lz / (float(self.part[2]-1))
        self.nodes_latt=np.empty(shape=self.part, dtype=tuple)
        self.nodes_latt_rot=np.empty(shape=self.part, dtype=tuple)

        u  = rot['vec']
        th = rot['angle']
        ux  = np.array([[0.0, -u[2], u[1]],
                        [u[2], 0.0, -u[0]],
                        [-u[1], u[0], 0.0]])
        uxu = np.outer(u, u)
        R = np.cos(th)*np.identity(3) + np.sin(th)*ux + (1.0-np.cos(th))*uxu
        for i in np.arange(self.part[0]):
            for j in np.arange(self.part[1]):
                for k in np.arange(self.part[2]):
                    v = np.dot(R, np.array([i*dx-Lx/2.0,
                                            j*dy-Ly/2.0,
                                            k*dz-Lz/2.0]))
                    self.nodes_latt_rot[i,j,k] = (v[0]+self.center[0],
                                                  v[1]+self.center[1],
                                                  v[2]+self.center[2])
                    self.nodes_latt[i,j,k] = (i*dx-Lx/2.0+self.center[0],
                                              j*dy-Ly/2.0+self.center[1],
                                              k*dz-Lz/2.0+self.center[2])

        # Convert lattice node structure into generic node structure
        self.nodeNum=np.zeros((self.part[0],self.part[1],self.part[2]),
                              dtype=int)
        self.nodes=np.zeros((self.part[0]*self.part[1]*self.part[2],3),
                            dtype=float)
        self.nodes_rot=np.zeros((self.part[0]*self.part[1]*self.part[2],3),
                                dtype=float)
        cnt=-1
        for k in np.arange(self.part[2]):
            for j in np.arange(self.part[1]):
                for i in np.arange(self.part[0]):
                    cnt+=1
                    self.nodes[cnt,:] = np.array( self.nodes_latt[i,j,k] )
                    self.nodes_rot[cnt,:] = np.array(self.nodes_latt_rot[i,j,k])
                    self.nodeNum[i,j,k] = cnt

    def get_nodes(self, deformed=False, x=None, vtype=None):
        """
        Return (deformed) nodes of Lattice object.
        """

        # Compute displacement if deformed lattice requested
        nodes = copy.copy(self.nodes_rot)
        if deformed:
            #print('in deformed')
            disp  =  self.deform(x, vtype)
            if self.rot is not None:
                u  = self.rot['vec']
                th = self.rot['angle']
                ux  = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])
                uxu = np.outer(u, u)
                R = np.cos(th)*np.identity(3) + np.sin(th)*ux + \
                    (1.0-np.cos(th))*uxu
                disp = np.dot(R, disp.T).T
            #print(np.linalg.norm(disp))
            nodes += disp
        return ( nodes )

    def add_degree_of_freedom(self, id, nodes_lattice, dirs, expr=None):
        """
        Add lattice degree of freedom.

        Parameters
        ----------
        id : int
        nodes : ndarray (or iterable)
          Node number of nodes whose dofs (all or some; specified in dir) will
          be lumped into a Design Variable
        dirs : list (len = nodes.size) of lists (len = 1, 2, 3) of int
          Degrees of freedom for each node that will be lumped into the current
          Design Variable
        expr : function
          Map from the vector of (all) Abstract Variables to the present
          Design Variable (returns scalar)

        Example
        -------
        >> obj.addDof(0,[(0,2,0),(2,0,1)],[1,1])
        Sets DOF 0 to the motion of node (0,2,0) in the y-direction and node
        (2,0,1) in the y-direction, i.e. they are constrained to move together.
        This function simply converts the Lattice description of the nodes
        (nx, ny, nz) into a global numbering (Fortran ordering:
        x-dimension = fastest varying).
        """

        # Striding to convert lattice nodes into general nodes
        nodes = np.array([x[0]+x[1]*self.part[0]+x[2]*self.part[0]*self.part[1]
                                                        for x in nodes_lattice])
        Deform.add_degree_of_freedom(self, id, nodes, dirs, expr)

    def make_degrees_of_freedom_from_nodes(self,dirs):
        """
        Loops over all nodes and adds between 1 and 3 DOFs per node (depending
        on which directions to convert to DOFs as specified in 'dirs'). Helper
        function to avoid doing this task manually for the common case of
        using all nodes as DOFs.

        Parameters
        ----------
        dirs : int or iterable of ints
          Directions to turn into degree of freedom at each node
        """

        # Ensure dirs is int or iterable of ints
        if not ( (type(dirs) is int) or
                          (hasattr(dirs,'__iter__') and type(dirs[0]) is int) ):
            raise ValueError('dirs must be int or iterable of ints')

        id = -1
        for k in np.arange(self.part[2]):
            for j in np.arange(self.part[1]):
                for i in np.arange(self.part[0]):
                     for dir in dirs:
                        id += 1
                        self.add_degree_of_freedom(id,[(i,j,k)],[dir])

    def blender_make_deform_object_from_self(self):
        """
        Make blender Lattice object from Lattice object. Only available during
        call to Blender.
        """

        rot = self.rot
        # Make blender lattice object
        latt = create_lattice(partitions=self.part, interp=self.interp)
        latt.location       = Vector(self.center)
        latt.scale          = Vector(self.scale)
        if rot is not None:
            vec = Vector(rot['vec'])
            mat = Matrix.Rotation(rot['angle'], 3, vec)
            latt.rotation_euler = mat.to_euler()
        #latt.rotation_euler = Euler (self.rot, 'XYZ')

        self.blend_objs.deform_obj  = latt
        self.blend_objs.deform_type = 'LATTICE'
        self.blend_objs.mesh_obj    = create_mesh_object('LATTICE', self.nodes)

        return ( latt )

    def blender_deform(self, x, vtype, **kwargs):
        """
        Apply deformation to deformee from values of node displacement, dsgvar,
        or absvar.  First, pass 'x' to deform pyAeroOpt Deform object, then
        impose this deformation on the blender modifiers.

        Parameters
        ----------
        x : ndarray
          Array containing nodal displacement, dsgvar, or absvar
        vtype : str
          Type of degree of freedom ('dof' for 'absvar' or 'dsgvar', 'nodes')
        """

        # Apply deformation to current Deform object (ensures self.disp has
        # appropriate values)
        rot = self.rot
        disp = self.deform(x, vtype, **kwargs)

        # Extract blender deform object and move nodes
        obj     = self.blend_objs.deform_obj
        #objMesh = self.blend_objs.mesh_obj
        for k, node in enumerate(self.nodes):
            new_point_scaled_trans = self.nodes[k,:] + disp[k,:]
            new_point = (new_point_scaled_trans - self.center)/self.scale
            obj.data.points[k].co_deform = Vector(tuple(new_point))

    def blender_extract_deformation(self):
        """
        Extract displacement form blender object.
        """
        disp = extract_object_displacement(self.blend_objs.mesh_obj)
        return ( disp )

    def get_connectivity(self):
        """
        Get connectivity of mesh
        """

        cnt = -1
        elems=np.zeros((self.part[0]*self.part[1]*(self.part[2]-1)+
                        self.part[0]*(self.part[1]-1)*self.part[2]+
                       (self.part[0]-1)*self.part[1]*self.part[2],2),dtype=int)
        for i in np.arange(self.part[0]):
            for j in np.arange(self.part[1]):
                for k in np.arange(self.part[2]-1):
                    cnt+=1
                    elems[cnt,:] = np.array((int(self.nodeNum[i,j,k]),
                                             int(self.nodeNum[i,j,k+1])))

        for i in np.arange(self.part[0]):
            for k in np.arange(self.part[2]):
                for j in np.arange(self.part[1]-1):
                    cnt+=1
                    elems[cnt,:] = np.array((int(self.nodeNum[i,j,k]),
                                             int(self.nodeNum[i,j+1,k])))

        for j in np.arange(self.part[1]):
            for k in np.arange(self.part[2]):
                for i in np.arange(self.part[0]-1):
                    cnt+=1
                    elems[cnt,:] = np.array((int(self.nodeNum[i,j,k]),
                                             int(self.nodeNum[i+1,j,k])))

        return ( elems )

    def write_top(self, fname, deformed=False, x=None, vtype=None):
        """
        Write top file for Lattice object. Option to write undeformed or
        deformed nodes.

        Parameters
        ----------
        fname : str
          Filename of top file
        deformed : bool
          Defines whether to add disp to nodes prior to writing nodes
        """

        # Get node and element names
        nodeSetName = 'LatticeNodes'
        elemType    = 106 # DO NOT CHANGE - XPOST & XP2EXO can read!

        # Get nodes and element
        nodes  = self.get_nodes(deformed, x, vtype)
        nnodes = nodes.shape[0]
        nnum   = np.arange(nnodes)+1

        elems  = self.get_connectivity()+1
        nelems = elems.shape[0]
        enum   = np.arange(nelems)+1
        etype  = elemType*np.ones(nelems,dtype=int)

        # Write top file
        write_top(fname, [['Nodes',nodeSetName,nnum,nodes]],
                         [['Elements','LattConn',nodeSetName,enum,etype,elems]])

    def write_xpost(self, fname, x, vtype):
        """
        Write xpost file for Lattice object.

        Parameters
        ----------
        fname : str
          Filename of xpost file
        """

        # Compute displacement
        disp = self.deform(x, vtype)
        print(np.linalg.norm(disp))
        if self.rot is not None:
            u  = self.rot['vec']
            th = self.rot['angle']
            ux  = np.array([[0.0, -u[2], u[1]],
                            [u[2], 0.0, -u[0]],
                            [-u[1], u[0], 0.0]])
            uxu = np.outer(u, u)
            R = np.cos(th)*np.identity(3) + np.sin(th)*ux + \
                (1.0-np.cos(th))*uxu
            print(R.shape)
            print(disp.shape)
            disp = np.dot(R, disp.T).T
        print(np.linalg.norm(disp))

        # Write to xpost file
        print(fname)
        write_xpost(fname, 'LatticeNodes', [0], disp, 'Displacement')

    def plot_object(self, deformed=False, x=None, vtype=None):
        """
        Plot Lattice object with matplotlib
        """

        # Extract input
        ax  = kwargs['ax']  if 'ax'  in kwargs else None
        dim = kwargs['dim'] if 'dim' in kwargs else '3D'
        con = kwargs['con'] if 'con' in kwargs else False
        nodeFormat = kwargs['nodeFormat'] if 'nodeFormat' in kwargs else {}
        conFormat  = kwargs['conFormat']  if 'conFormat'  in kwargs else {}

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Generate figure/axes
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')

        # Extract nodes and plot
        nodes = self.get_nodes(deformed, x, vtype)
        ax.plot(nodes[:,0],nodes[:,1],nodes[:,2],**nodeFormat)

        # Extract elements and plot
        if con:
            elems = self.get_connectivity()
            for elem in elems:
                ax.plot(nodes[[elem[0],elem[1]],0],
                        nodes[[elem[0],elem[1]],1],
                        nodes[[elem[0],elem[1]],2],**conFormat)
        plt.show()

class MeshDeform(Deform):
    """
    MeshDeform object
    """
    def __init__(self, id, nodes, edges=[], faces=[], normals=[], **kwargs):
        """
        Constructor. Do not specify both edges and faces!

        Parameters
        ----------
        id : int
        nodes : ndarray
          nnode x 3 array of nodes of mesh
        edges : list of tuples
          Edges of mesh
        faces : list of tuples
          Faces of elements of mesh (heterogeneous mesh supported)
        """

        # Inputs/defaults
        Deform.__init__(self,id)

        self.nodes = nodes
        #[(x[0],x[1],x[2]) for x in nodes]
        self.edges = edges
        self.faces = faces
        self.normals = normals

        self.weight    = kwargs['weight']    if 'weight'    in kwargs else 1.0
        self.precision = kwargs['precision'] if 'precision' in kwargs else 4
        self.dynamic   = kwargs['dynamic']   if 'dynamic'   in kwargs else False

    def blender_make_deform_object_from_self(self):
        """
        Make blender MeshDeform object from MeshDeform object. Only available
        during call to Blender.
        """
        cage = create_mesh_deform( 'cage-'+str(self.id),self.nodes,
                                             self.edges,self.faces,self.normals)
        self.blend_objs.deform_obj=cage
        self.blend_objs.deform_type='MESH_DEFORM'
        return cage

    def blender_apply_deform_object_to_mesh_object(self,ob):
        """
        Apply blender modifier to blender object. Only available during call to
        Blender.
        """
        mod=Deform.blender_apply_deform_object_to_mesh_object(self,ob,
                                  precision=self.precision,dynamic=self.dynamic)
        bindCage(ob,mod)

    def blender_deform(self, x, vtype, **kwargs):
        """
        Apply deformation to deformee from values of node displacement, dsgvar,
        or absvar.  First, pass 'x' to deform pyAeroOpt Deform object, then
        impose this deformation on the blender modifiers.

        Parameters
        ----------
        x : ndarray
          Array containing nodal displacement, dsgvar, or absvar
        vtype : str
          Type of degree of freedom ('dof' for 'absvar' or 'dsgvar', 'nodes')
        """

        # Apply deformation to current Deform object (ensures self.disp has
        # appropriate values)
        disp = self.deform(x, vtype, **kwargs)

        # Extract blender deform object and move nodes
        obj = self.blend_objs.deform_obj
        #objMesh = self.blend_objs.mesh_obj
        for k, node in enumerate(self.nodes):
            newPoint = self.nodes[k,:] + disp[k,:]
            obj.data.vertices[k].co = Vector(tuple(newPoint))
            #objMesh.data.vertices[k].co = Vector(tuple(newPoint))


    def get_connectivity(self):
        """
        Get connectivity of mesh
        """

        # Elements
        nTri = 0
        for f in self.faces: nTri += ( len(f) - 2 )
        elems = np.zeros((nTri,3),dtype=int)

        cnt=-1
        for k, face in enumerate(self.faces):
            norm     = self.normals[k]
            for t, tri in enumerate(face[2:]):
                cnt+=1
                ind=t+2
                elems[cnt,:]=np.array([face[0],face[ind-1],tri],dtype=int)

                calc = np.cross(self.nodes[face[ind-1]]-self.nodes[face[0]],
                                self.nodes[tri        ]-self.nodes[face[0]])
                if np.dot(norm,calc) < 0.0:
                    elems[cnt,1] = tri
                    elems[cnt,2] = face[ind-1]

        return ( elems )

    def blender_extract_deformation(self):
        """
        Extract displacement form blender object.
        """
        disp = extract_object_displacement(self.blend_objs.deform_obj)
        return ( disp )

    def write_top(self, fname, deformed=False, x=None, vtype=None):
        """
        Write top file for Cage object. Option to write undeformed or deformed
        nodes.

        Parameters
        ----------
        fname : str
          Filename of top file
        deformed : bool
          Defines whether to add disp to nodes prior to writing nodes
        """

        # Get node and element names
        nodeSetName = 'CageNodes'
        elemType    = 4 # DO NOT CHANGE - XPOST & XP2EXO can read!

        # Get nodes and element
        nodes  = self.get_nodes(deformed, x, vtype)
        nnodes = nodes.shape[0]
        nnum   = np.arange(nnodes)+1

        elems  = self.get_connectivity()+1
        nelems = elems.shape[0]
        enum   = np.arange(nelems)+1
        etype  = elemType*np.ones(nelems,dtype=int)

        # Write top file
        write_top(fname, [['Nodes',nodeSetName,nnum,nodes]],
                         [['Elements','CageConn',nodeSetName,enum,etype,elems]])

    def write_xpost(self, fname, x, vtype):
        """
        Write xpost file for Cage object.

        Parameters
        ----------
        fname : str
          Filename of xpost file
        """

        # Compute displacement
        disp = self.deform(x, vtype)

        # Write to xpost file
        write_xpost(fname, 'CageNodes', [0], disp, 'Displacement')

    def plot_object(self, deformed=False, x=None, vtype=None):
        pass

class Armature(Deform):
    def __init__(self,id,nodes,edges,nseg=[],radii=[],env=[],**kwargs):
        # Inputs/defaults
        Deform.__init__(self,id)

        self.nodes = np.array([np.array(node) for node in nodes])
        #self.nodes = [(x[0],x[1],x[2]) for x in nodes]
        self.edges = edges
        self.nseg  = nseg
        self.radii = radii
        self.env   = env

        self.weight    = kwargs['weight'] if 'weight' in kwargs else 1.0

    def add_degree_of_freedom(self,id,elems,dirs,expr=None):

        # Ensure nodes iterable and dirs list of lists
        if not hasattr(elems,'__iter__'):
            raise ValueError('elems argument must be iterable')
        if type(dirs) is not list:
            raise ValueError('dirs argument must be list of lists. If dirs[i] '+
                             'only needs to be a scalar, it will be converted '+
                             'to list internally.')
        else:
            dirs = [x if type(x) is list else [x] for x in dirs]

        # Add degree of freedom
        self.dofs.add_to_group(Object(id, 'DOF', elems=elems,
                                      dirs=dirs, expr=expr) )

    def move_degrees_of_freedom(self,x):
        pass

    def blender_make_deform_object_from_self(self):
        """
        Make blender Armature object from Armature object. Only available during
        call to Blender.
        """
        # Make blender mesh deform object
        #arm = createArmature('arm','ENVELOPE')
        arm = createArmature('arm','OCTAHEDRAL')
        createBonesInArmature('bone',arm,self.nodes,self.edges,self.nseg,
                                                            self.radii,self.env)
        self.blend_objs.deform_obj=arm
        self.blend_objs.deform_type='ARMATURE'
        return ( arm )

    def blender_deform(self, x, vtype, **kwargs):
        """
        Apply deformation to deformee from values of node displacement, dsgvar,
        or absvar.  First, pass 'x' to deform pyAeroOpt Deform object, then
        impose this deformation on the blender modifiers.

        Parameters
        ----------
        x : ndarray
          Array containing nodal displacement, dsgvar, or absvar
        vtype : str
          Type of degree of freedom ('dof' for 'absvar' or 'dsgvar', 'nodes')
        """

        # Ensure in POSE mode
        arm = self.blend_objs.deform_obj
        bpy.context.scene.objects.active = arm
        bpy.ops.object.mode_set(mode='POSE')

        # Apply deformation to current Deform object (ensures self.disp has
        # appropriate values)
        for d, dof in enumerate(self.dofs.get_from_id()):

            # Extract value for degree of freedom
            if self.vtype == 'absvar':
                val = dof.expr(x)
            else:
                val = x[d]

            # Rotation matrices
            rotMats = [ Matrix.Rotation(val,3,'X'),
                        Matrix.Rotation(val,3,'Y'),
                        Matrix.Rotation(val,3,'Z') ]

            # Impose appropriate rotation matrices and apply to bones
            for j, enum in enumerate(dof.elems):
                rotMatrix = Matrix.Translation((0.0,0.0,0.0)).to_3x3()
                for k, dir in enumerate(dof.dirs[j]):
                    rotMatrix.rotate( rotMats[k] )    
                bone = arm.pose.bones['bone'+str(enum).zfill(3)]
                bone.rotation_quaternion = rotMatrix.to_quaternion()

        # Ensure in OBJECT mode
        bpy.ops.object.mode_set(mode='OBJECT')

###############################################################################

def run_code(x, deform_obj, ptcloud, vmo, der=None, eps=1.0e-8, xpost=None,
             log='blender.tmp.log', make_call=True, blender_exec=None,
             bg=True, mpiexec=None, nproc=1):
    """
    Pickle input arguments and run Blender.
    """

    # Pickle *args for unpickling inside blender call (don't change fnamePickle
    # unless you change it in blender_deform.py as well)
    import pickle
    pkf = open(pickle_filename(), 'wb')
    pickle.dump([x, deform_obj, ptcloud, vmo, der, eps, xpost], pkf) 
    pkf.close()

    # Default blender executable
    if blender_exec is None: blender_exec = os.path.expandvars('$BLENDER')

    # Build execution string
    exec_str = blender_exec
    if mpiexec is not None and nproc > 1:
        exec_str = "{0:s} -np {1:d} {2:s}".format(mpiexec, nproc, exec_str)
    if bg: exec_str+=' -b'
    exec_str+=(' -P '+os.path.expandvars('$PYAEROOPT')+
                                             'pyaeroopt/util/blender/deform.py')

    # Call Blender and clean directory
    execute_code(exec_str, log, make_call)
    clean_directory()

def pickle_filename():
    return 'tmpFileForPassingClassesToBlenderViaPickle.pkl'

def clean_directory():
    subprocess.call('rm {0:s}'.format(pickle_filename()), shell=True)    
