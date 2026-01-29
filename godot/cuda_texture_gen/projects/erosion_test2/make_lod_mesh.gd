"""
"""
@tool
extends MeshInstance3D



@export var trigger_action := false:
    set(value):
        if value:
            do_something()
            trigger_action = false  # Reset immediately



func generate_planar_uvs(vertices: PackedVector3Array, total_size := Vector2(1.0, 1.0)) -> PackedVector2Array:
    var uvs := PackedVector2Array()
    uvs.resize(vertices.size())

    for i in vertices.size():
        var v := vertices[i]
        var u := v.x / total_size.x
        var w := v.z / total_size.y
        uvs[i] = Vector2(u, w)
    return uvs


func build_grid(stc, total_size: Vector2, subdiv: Vector2i, position: Vector3 = Vector3.ZERO):
    var sx = subdiv.x
    var sz = subdiv.y

    var dx = total_size.x / float(sx)
    var dz = total_size.y / float(sz)

    for x in range(sx):
        for z in range(sz):
            var x0 = x * dx
            var z0 = z * dz
            var x1 = x0 + dx
            var z1 = z0 + dz

            # Build local quad
            var verts := PackedVector3Array([
                Vector3(x0, 0, z0),
                Vector3(x1, 0, z0),
                Vector3(x1, 0, z1),
                Vector3(x0, 0, z1),
            ])

            # Apply world-space position offset
            for i in verts.size():
                verts[i] += position

            # Planar UVs based on local coordinates (not world)
            var uvs := generate_planar_uvs(verts, total_size)

            stc.make_ngon(verts, uvs)



@export var ring_order := 2

# builds a ring of squares, 0 is one square, 1 is surrounding squares and so on
func build_square_ring(stc, macro_size := Vector2(16, 16), subdiv := Vector2i(8,8), order = 2, offset := Vector3.ZERO):
    
    
    if order == 0:
        build_grid(stc, macro_size, subdiv, offset)
        return
    
    for i in order * 2:
        # N
        var seg_pos := Vector2i(-order + 1 + i, order)        
        var world_pos = Vector3(macro_size.x, 0, macro_size.y) * Vector3(seg_pos.x, 0, seg_pos.y)
        build_grid(stc, macro_size, subdiv, world_pos + offset)
        # E
        seg_pos = Vector2i(-seg_pos.y, seg_pos.x)
        world_pos = Vector3(macro_size.x, 0, macro_size.y) * Vector3(seg_pos.x, 0, seg_pos.y)
        build_grid(stc, macro_size, subdiv, world_pos + offset)
        # S
        seg_pos = Vector2i(-seg_pos.y, seg_pos.x)
        world_pos = Vector3(macro_size.x, 0, macro_size.y) * Vector3(seg_pos.x, 0, seg_pos.y)
        build_grid(stc, macro_size, subdiv, world_pos + offset)
        # W
        seg_pos = Vector2i(-seg_pos.y, seg_pos.x)
        world_pos = Vector3(macro_size.x, 0, macro_size.y) * Vector3(seg_pos.x, 0, seg_pos.y)
        build_grid(stc, macro_size, subdiv, world_pos + offset)



@export var lod_segment_size := Vector2(16, 16)
@export var lod_subdiv := Vector2i(16, 16)



#@export var lod_ring_div := [128, 128, 64, 32, 16, 8, 4, 2, 1]
#@export var lod_ring_div := [128, 128, 64, 64, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 8, 4, 4, 4]
@export var lod_ring_div := [128, 128, 64, 64]



func build_lod_mesh_type3(stc: SurfaceToolCache) -> void:
    

    var offset := -Vector3(lod_segment_size.x, 0, lod_segment_size.y) / 2.0
    
    for i in lod_ring_div.size():        
        build_square_ring(stc, lod_segment_size, Vector2i(1,1) * lod_ring_div[i], i, offset)
        
    
func calculate_lod_mesh_type3_triangle_count() -> int:
    
    var triangles = 0
    
    for i in lod_ring_div.size():
        var div = lod_ring_div[i]
        
        if i == 0:
            triangles += div * div * 2
        else:
            var count = div * div * 2 # per square
            count *= i * 8
            triangles += count
    
    
    print("estimated: ", triangles)
    return triangles


func do_something():
    
    print("do_something()...")
    
    calculate_lod_mesh_type3_triangle_count()
    
    #var stc := SurfaceToolCache.new()
    #build_lod_mesh_type3(stc)
    #mesh = stc.get_mesh()
    
    
    
    
    
    
    
