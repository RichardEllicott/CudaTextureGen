"""
"""
@tool
extends MeshInstance3D


@export var trigger_update := false:
    set(value):
        if value:
            do_something()
            trigger_update = false  # Reset immediately
           


static var DEFAULT_VERTS := PackedVector3Array([
    Vector3(0,0,0),
    Vector3(1,0,0),
    Vector3(1,0,1),
    Vector3(0,0,1),
])

# we can 
static var DEFAULT_UVS := PackedVector2Array([
    Vector2(0,0),
    Vector2(1,0),
    Vector2(1,1),
    Vector2(0,1),
])

func get_default_uvs() -> PackedVector2Array:
    return PackedVector2Array([
    Vector2(0,0),
    Vector2(1,0),
    Vector2(1,1),
    Vector2(0,1),
])


func get_default_verts() -> PackedVector3Array:
    return PackedVector3Array([
        Vector3(0,0,0),
        Vector3(1,0,0),
        Vector3(1,0,1),
        Vector3(0,0,1),
    ])
    



func build_edge_01(stc: SurfaceToolCache):
    
    var verts := DEFAULT_VERTS
    var uvs := DEFAULT_UVS
    
    stc.make_ngon(verts, uvs)
    
    # Rotate 90 degrees around X so the quad points downward
    var rot = Basis(Vector3(1,0,0), deg_to_rad(-90))
    #var rot = Basis(Vector3(1,0,0), deg_to_rad(-45))
    for i in verts.size():
        verts[i] = verts[i] * rot
    
    verts = GeometryUtils.reverse_verts(verts)
    verts = GeometryUtils.cycle_verts(verts, 2)
    
    stc.make_ngon(verts, uvs)




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




    

func build_bevel_edge_01(stc: SurfaceToolCache):
    
    
    var verts := get_default_verts()
    var uvs := DEFAULT_UVS
    
    #verts = PackedVector3Array([
        #Vector3(0,0,0),
        #Vector3(1,0,0),
        #Vector3(1,1,0),
        #Vector3(0,1,0),
        #])

    verts = PackedVector3Array([
        Vector3(0,1,0),
        Vector3(1,1,0),
        Vector3(1,0,0),
        Vector3(0,0,0),
        ])
    
    var quads = GeometryUtils.split_and_extrude(verts, uvs, 0, 0.25, 0.25)
    
    for quad in quads:
        stc.make_ngon(quad.verts, quad.uvs)


func do_something():
    
    var stc := SurfaceToolCache.new()
    
    #build_edge_01(stc)
    build_bevel_edge_01(stc)
    
    mesh = stc.get_mesh()
            
            
            
