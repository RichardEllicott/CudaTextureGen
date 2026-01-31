"""
"""
@tool
extends MeshInstance3D


@export var trigger_update := false:
    set(value):
        if value:
            _trigger_update()
            trigger_update = false  # Reset immediately
           
@export var generate_normal_texture := false

@export var mode := 0



func get_default_verts() -> PackedVector3Array:
    return PackedVector3Array([
        Vector3(0,0,0),
        Vector3(1,0,0),
        Vector3(1,0,1),
        Vector3(0,0,1),
    ])
    


func get_default_uvs() -> PackedVector2Array:
    return PackedVector2Array([
    Vector2(0,0),
    Vector2(1,0),
    Vector2(1,1),
    Vector2(0,1),
])

func test_build_edge_01(stc: SurfaceToolCache) -> void:
    
    var offset = Vector3(0,1,0)
    #var offset := Transform3D.IDENTITY
    #offset.origin = Vector3(0,1,0)
    
    var verts := get_default_verts()
    var uvs := get_default_uvs()
        
    GeometryUtils.transform_verts(verts, offset)
    stc.add_ngon(verts, uvs)
    
    verts = get_default_verts()    
    var rot = Basis(Vector3(1,0,0), deg_to_rad(90))
    GeometryUtils.transform_verts(verts, rot)        
    GeometryUtils.transform_verts(verts, offset)        

    #verts = GeometryUtils.reverse_verts(verts)
    verts.reverse()
    
    verts = GeometryUtils.cycle_verts(verts, 2)
    
    stc.add_ngon(verts, uvs)
    
    if generate_normal_texture:
        
        var material = get_surface_override_material(0)
        
        if not material is StandardMaterial3D:
            material = StandardMaterial3D.new()
        
        material.normal_enabled = true
        
        var image = NormalMapUtils.generate_bevel_normal_map(256, 256, 32, 0, 32, 0)
        var texture = ImageTexture.create_from_image(image)
        
        material.normal_texture = texture
        
        set_surface_override_material(0, material)

func test_build_bevel_edge_01(stc: SurfaceToolCache) -> void:
    
    
    var verts := get_default_verts()
    var uvs := get_default_uvs()
    
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
        stc.add_ngon(quad.verts, quad.uvs)

func test_build_tube(stc: SurfaceToolCache) -> void:
    

    var height = 1.0
    height = PI / 2.0
    var quads = GeometryUtils.get_tube_quads(0.25, height, 16, 16, false, Vector2(2,2))
    stc.smooth_group = 0
    for quad in quads:
        GeometryUtils.simple_deform_bend_points(quad.verts, 90.0, height)
        stc.add_ngon(quad.verts, quad.uvs)


func test_build_cube(stc: SurfaceToolCache) -> void:
    
    stc.smooth_group = -1
    
    var verts := PackedVector3Array([
        Vector3(1, 1, 0), 
        Vector3(0, 1, 0), 
        Vector3(0, 0, 0), 
        Vector3(1, 0, 0)])


    var uvs := get_default_uvs()
    
    #GeometryUtils.transform_verts(verts, Vector3(0, 1, 0))
#
    #
    for quad in GeometryUtils.extrude_ngon(verts, uvs, 1.0):
        quad.uvs = GeometryUtils.get_cube_projected_uvs(quad.verts)        
        #quad.verts.reverse()
        stc.add_ngon(quad.verts, quad.uvs)
    



func _trigger_update() -> void:
    
    var stc := SurfaceToolCache.new()
    stc.smooth_group = -1
    
    match mode:
        0: test_build_edge_01(stc)
        1: test_build_bevel_edge_01(stc)
        2: test_build_tube(stc)
        3: test_build_cube(stc)

    mesh = stc.get_mesh()
            
            
            
