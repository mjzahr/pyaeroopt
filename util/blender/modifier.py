
import bpy
from pyaeroopt.util.blender.mesh import create_mesh_object

def add_modifier(ob, mod, mtype, vert_grp_name='', **kwargs):
    """
    Add Deform Modifier as Modifier to Object.
   
    Parameters
    ----------
    ob : Object
    mod : Deform Modifier
    mtype : str
      Type of modifier to be added. Options include
      'LATTICE'/'ARMATURE'/'MESH_DEFORM'/'LAPLACIANDEFORM'/'SIMPLE_DEFORM'/
      'CAST'/'CURVE'/'DISPLACE'/'HOOK'/'LAPLACIANSMOOTH'/'SHRINKWRAP'/
      'SMOOTH'/'WARP'/'WAVE'
    vert_grp_name : str
      Name of vertex group specifying the vertices to which the modifier will
      be applied, use '' for all vertices
    """

    # Ensure in object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add modifier to scene and extract
    bpy.context.scene.objects.active = ob
    bpy.ops.object.modifier_add(type=mtype)
    mod_obj = ob.modifiers[-1]
    mod_obj.object = mod

    # For mesh deform modifier, set precision
    if mtype=='MESH_DEFORM':
        # Precision of binding (higher precision can be very slow and cause
        # Blender to crash for large meshes)
        precision = kwargs['precision'] if 'precision' in kwargs else 4
        mod_obj.precision = precision

        # Dynamic binding (ensures points stay in cage when multiple modifiers
        # applied)
        dynamic = kwargs['dynamic']   if 'dynamic'   in kwargs else False
        mod_obj.use_dynamic_bind = dynamic

    elif mtype=='ARMATURE':
        # Use envelopes to deform (radius of influence)
        envelopes  = kwargs['envelopes']   if 'envelopes'   in kwargs else True
        mod_obj.use_bone_envelopes = envelopes

        # Perform rotation interpolation with quaternions
        preserveVol= kwargs['preserveVol'] if 'preserveVol' in kwargs else True
        mod_obj.use_deform_preserve_volume = preserveVol

        # Don't use vertex groups
        mod_obj.use_vertex_groups = False

        # For use with multiple armature modifier levels
        mod_obj.use_multi_modifier  = False

        # Invert influence of vert
        mod_obj.invert_vertex_group = False

    if (not vert_grp_name): mod_obj.vertex_group = vert_grp_name
    bpy.context.scene.update()
    return ( mod_obj )

#################################### LATTICE ###################################
def create_lattice(partitions=(4,4,4), interp="Bspline"):
    """
    Generate a lattice for the Lattice modifer.

    Parameters
    ----------
    interp : str
      Interpolation between lattice points used for FFD.  Options include
      "Linear"/"Cardinal"/"Bspline"
    partitions : tuple of int ([0, 64])
      Number of control points in each direction
    """

    bpy.ops.object.add(type='LATTICE')
    latt = bpy.context.scene.objects.active

    latt.data.points_u = partitions[0]
    latt.data.points_v = partitions[1]
    latt.data.points_w = partitions[2]
    if interp.lower() == "linear":
        mode = 'KEY_LINEAR'
    elif interp.lower() == "cardinal":
        mode = 'KEY_CARDINAL'
    elif interp.lower() == "bspline":
        mode = 'KEY_BSPLINE'
    elif interp.lower() == "catmull":
        mode = 'KEY_CATMULL_ROM'
    else:
        print('Invalid lattice mode ', interp,'for AutoGenerateLattice')
        return
    latt.data.interpolation_type_u = mode
    latt.data.interpolation_type_v = mode
    latt.data.interpolation_type_w = mode

    return(latt)

################################# MESH DEFORM ##################################
def create_mesh_deform(name, verts, edges=[], faces=[], normals=[]):
    """
    Generate cage for the MESHDEFORM modifier.
    """
    if len(edges) == 0 and len(faces) == 0:
        raise ValueError('Either edges or faces must be non-empty to create '+
                                                                  'closed mesh')
    # Create blender object from mesh
    cage = create_mesh_object(name, verts, edges, faces, normals)

    # Make cage easy to see relative to mesh (only relevant if blender GUI used)
    cage.show_wire = True
    cage.show_transparent = True
    cage.draw_type = 'SOLID'
    
    mat = bpy.data.materials.new('Cage')
    mat.diffuse_color     = (0.0,0.1,1.0)
    mat.diffuse_intensity = 1.0
    mat.emit              = 0.5

    mat.alpha             = 0.2
    mat.use_transparency  = True

    # Make cage active object
    bpy.context.scene.objects.active = cage
    bpy.context.object.data.materials.append(mat)

    return cage

def bindCage(ob, mod):
    """
    Bind cage to an object.

    Parameters
    ----------
    ob : Blender Object
    mod : Blender MESHDEFORM modifier
    """

    if mod.is_bound:
        print("Cage already bound")
        return
    elif mod.precision > 4 and len(ob.data.vertices) > 5000:
        print("Computing cage bindings at precision %s on large mesh." %
                                                                (mod.precision))
        print("This may take awhile (or crash)")

    bpy.context.scene.objects.active = ob
    bpy.ops.object.meshdeform_bind(modifier=mod.name)

################################### ARMATURE ###################################
## Automatically generate a Armature modifer
#
def createArmature(name, atype='ENVELOPE'):
    """
    Generate Armature modifier.

    Parameters
    ----------
    name : str
      Name to assign to modifier
    atype : str
      Armature modifier behavior
    """

    bpy.ops.object.armature_add(location=(0.0,0.0,0.0),rotation=(0.0,0.0,0.0))
    arm = bpy.context.object

    arm.data.draw_type = atype
    arm.show_x_ray = True

    bpy.ops.object.mode_set(mode='EDIT')

    return(arm)

def create_bones_in_armature(name, arm, verts, edges, nseg, radii, envelope):

    # Ensure in edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    bones = []
    # Add bones to armature
    for k, edge in enumerate(edges):

        # Add bone to armature
        if k == 0:
            # Use root bone
            bones.append(arm.data.edit_bones[0])
            bones[-1].name = name+str(k).zfill(3)
        else:
            bones.append(arm.data.edit_bones.new(name+str(k).zfill(3)))

        # Create head of bone
        bones[k].head        = verts[edge[0]]
        bones[k].head_radius = radii[k][0]

        # Create tail of bone
        bones[k].tail        = verts[edge[1]]
        bones[k].tail_radius = radii[k][1]

        # Add number of segments (BBones)
        bones[k].bbone_segments = nseg[k]

        # Add envelope distance
        bones[k].envelope_distance = envelope[k]

    # Associate bones
    for k, kEdge in enumerate(edges):
        for j, jEdge in enumerate(edges):
            if k == j: continue
            if kEdge[0] == jEdge[1]:
                bones[k].parent = bones[j]

    # Add properties to bones in armature
    for k, edge in enumerate(edges):
        # Properties
        bones[k].roll                 = 0
        bones[k].use_connect          = True
        bones[k].use_deform           = True
        bones[k].use_inherit_rotation = True
        bones[k].use_inherit_scale    = False
        bones[k].use_local_location   = False

    bpy.ops.object.mode_set(mode='OBJECT')

    return (bones)    

############################## LAPLACIAN DEFORM ################################
def createLaplacianDeform():
    """
    Generate LaplacianDeform modifier.
    """
    pass

################################ SIMPLE DEFORM #################################
def createSimpleDeform():
   """
   Generate SimpleDeform modifier.
   """
   pass
