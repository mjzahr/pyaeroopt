import bpy

def delete_object(*objects):
    """
    Delete object and all children
    """

    for ob in objects:
      for child in ob.children:
          delete_object(child)

      if ob.name not in bpy.context.scene.objects:
         return
      ob.hide = False
      bpy.context.scene.objects.unlink(ob)

def delete_all_objects():
    """
    Delete all objects in active scene
    """

    while (bpy.context.scene.objects):
        delete_object(bpy.context.scene.objects[0])

def register():
    pass
def unregister():
    pass
