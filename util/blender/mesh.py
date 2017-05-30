from numpy import array

import bpy
from mathutils import Vector

def create_mesh_object(name, verts, edges=[], faces=[], normals=[]):
    """
    Create blender mesh as blender object from data (vertices,edes, faces).

    Parameters
    ----------
    verts : list of 3-tuple
      XYZ coordinates of each vertex
    edges : list of tuple
      Node numbers of edges
    faces : list of tuple
      Edge numbers of faces
    """

    if name in bpy.data.objects:
        bpy.data.objects[name].name = 'overwritten'
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)

    ob = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.objects.link(ob)
    bpy.context.scene.update()

    ob.select = True
    bpy.context.scene.objects.active = ob

    if len(normals) > 0:
        # deselect everything
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action = 'DESELECT')
    
        # enter object mode and reselect polygons that need to be flipped
        # (only those with incorrect normals)
        bpy.ops.object.mode_set()
        for k, poly in enumerate(ob.data.polygons):
            dp = poly.normal.dot(Vector(tuple(normals[k])))
            if dp < 0.0:
                poly.select = True

        # enter edit mode to flip normals
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.flip_normals()

        # finally, enter object mode again to maintain expected context
        bpy.ops.object.mode_set()

    return ( ob )

def create_mesh(name, verts, edges=[], faces=[]):
    """
    Create blender mesh from data (vertices, edges, faces).

    Parameters
    ----------
    verts : list of 3-tuple
      XYZ coordinates of each vertex
    edges : list of tuple
      Node numbers of edges
    faces : list of tuple
      Edge numbers of faces
    """

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    return mesh

def extract_mesh(ob):
    """
    Extract mesh with all modifiers applied in global coordinates.

    Parameters
    ----------
    ob : Blender Mesh object
    """

    bpy.context.scene.update()
    mesh = ob.to_mesh(scene=bpy.context.scene,
                      apply_modifiers=True,
                      settings='PREVIEW')
    mesh.transform(ob.matrix_world)
    return mesh

def extract_object_displacement(ob):
    """
    Extract deformation of object

    Parameters
    ----------
    ob : Blender Object
    """

    bpy.context.scene.update()

    # Create mesh from deformed and undeformed object
    mesh_undef = ob.to_mesh(scene=bpy.context.scene,
                            apply_modifiers=False,
                            settings='PREVIEW')
    mesh_def   = extract_mesh(ob)
    mesh_def.transform(ob.matrix_world)

    # Extract displacment and convert to numpy array
    undef_nodes = array( [ array( z.co ) for z in mesh_undef.vertices ] )
    def_nodes   = array( [ array( z.co ) for z in mesh_def.vertices   ] )
    disp = def_nodes - undef_nodes

    # Free the meshes
    bpy.data.meshes.remove( mesh_undef )
    bpy.data.meshes.remove( mesh_def )

    return ( disp )

def register():
    pass

def unregister():
    pass
