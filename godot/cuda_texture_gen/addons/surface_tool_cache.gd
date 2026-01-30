"""

simple cache for SurfaceTool, handles automatic UVs (cube projection)

usage example:
    
var verts = [
    Vector3(-1.0, 0.0, -1.0),
    Vector3(1.0, 0.0, -1.0,),
    Vector3(1.0, 0.0, 1.0,),
    Vector3(-1.0, 0.0, 1.0,)
    ]

var st_cache := SurfaceToolCache.new() # create a cache
st_cache.make_ngon(verts) # make ngon with automatic uvs
mesh = st_cache.get_mesh() # get mesh, will have automatic uvs

# can also set the material_index, to allow multiple surfaces (use the surface material override)

does not set materials, set these after creating the mesh on on material override
    
    
30/01/2026

added static helpers
    
"""
@tool
class_name SurfaceToolCache


## uv scale for cube projection
var uv_scale := Vector2(1.0, 1.0)
## uv offset for cube projection
var uv_offset := Vector2.ZERO
## material_index will trigger more surface tools to be created, whih allows multiple materials for mesh
var material_index := 0
## set to -1 for flat shading, otherwise smooth shading will be used
var smooth_group := 0
## correct reversed textures so no textures are mirrored
var correct_reversed_uvs = true

## private surface tools
var _surface_tools: Array[SurfaceTool] = []

var debug_print := true


enum UVMode {
    Default,
    CubeProject,
    XProject,
    YProject,
    ZProject
}

# default uv projection mode (if we don't supply a uv)
var uv_mode:= UVMode.Default

static var DEFAULT_QUAD_UVS = PackedVector2Array([
    Vector2(0,0),
    Vector2(1,0),
    Vector2(1,1),
    Vector2(0,1),
])
    



## private, get the surface tool based on material_index
func _get_surface_tool() -> SurfaceTool:
    
    while _surface_tools.size() <= material_index:
        var st := SurfaceTool.new()
        st.begin(Mesh.PRIMITIVE_TRIANGLES)
        _surface_tools.append(st)
        
    return _surface_tools[material_index]

## get the final mesh, generates normals and tangents
func get_mesh() -> Mesh:
    var mesh: Mesh
    
    for st: SurfaceTool in _surface_tools:
                
        st.generate_normals()
        #st.deindex()
        st.index() # found need to index here after generating normals
        st.generate_tangents()
        
        mesh = st.commit(mesh)
    
    if debug_print:
    
        var mesh_data = mesh.surface_get_arrays(0)
        var vertices = mesh_data[Mesh.ARRAY_VERTEX]
        print("Mesh Final vertex count: ", vertices.size())
        
        var indices = mesh_data[Mesh.ARRAY_INDEX]
        if indices:
            print("Index count: ", indices.size())
            print("Triangle count: ", indices.size() / 3)
    
    return mesh

## determine dominant axis of a normal for cube project
static func get_axis(input: Vector3) -> int:
    
    input.x = abs(input.x) 
    input.y = abs(input.y)
    input.z = abs(input.z)
    
    if input.x >= input.y:
        if input.x >= input.z:
            return Vector3.AXIS_X
        else:
            return Vector3.AXIS_Z
    else:
        if input.y >= input.z:
            return Vector3.AXIS_Y
        else:
            return Vector3.AXIS_Z

## get normal direction based on first three verts for cube project
static func get_normal(vertices: PackedVector3Array) -> Vector3:
    assert(vertices.size() >= 3)
    var v1 := vertices[1] - vertices[0]
    var v2 := vertices[2] - vertices[0]
    return v1.cross(v2).normalized()



    

## generate uvs with cube projection (or now other settings)
func generate_uvs(vertices: PackedVector3Array) -> PackedVector2Array:
    
    assert(vertices.size() >= 3)

    var uvs := PackedVector2Array() # build uv array
    uvs.resize(vertices.size()) # set it's size
    
    var v1 := vertices[1] - vertices[0]
    var v2 := vertices[2] - vertices[0]
    var cross := v1.cross(v2)
    
    var axis: int
    
    match uv_mode:
        UVMode.Default:
            for i in uvs.size():
                uvs[i] = DEFAULT_QUAD_UVS[i] * uv_scale + uv_offset
            return uvs
                
        UVMode.CubeProject:
            axis = get_axis(cross)
        UVMode.XProject:
            axis = Vector3.AXIS_X
        UVMode.YProject:
            axis = Vector3.AXIS_Y
        UVMode.ZProject:
            axis = Vector3.AXIS_Z
            
    
    
    
    for i in vertices.size():    
        var vert := vertices[i]
        var uv := Vector2.ZERO
        
        match axis:
            Vector3.AXIS_X:
                uv = Vector2(-vert.z, -vert.y)
            Vector3.AXIS_Y:
                uv = Vector2(vert.x, vert.z)
            Vector3.AXIS_Z:
                uv = Vector2(vert.x, -vert.y)
        
        # reversed uv correction
        if correct_reversed_uvs and cross[axis] > 0.0:
            uv.x = -uv.x
                    
        uvs[i] = uv * uv_scale + uv_offset

    return uvs

## make an ngon, if the uvs is empty or wrong size, we will build it with a cube projection
func make_ngon(
    vertices: PackedVector3Array,
    uvs: PackedVector2Array = PackedVector2Array(),
    colors: PackedColorArray = PackedColorArray()
    ) -> void:
        
    assert(vertices.size() >= 3)
    
    # if uvs empty, we assume we want a cube projection
    if vertices.size() != uvs.size():
        uvs = generate_uvs(vertices)
    else:
        # if we are using custom uvs but have some non-default scaling settings
        if uv_scale != Vector2(1.0, 1.0) or uv_offset != Vector2.ZERO:            
            uvs = uvs.duplicate() # make a copy so we avoid editing the orginal
            for i in uvs.size():
                uvs[i] = uvs[i] * uv_scale + uv_offset
    
    _get_surface_tool().set_smooth_group(smooth_group)
    _get_surface_tool().add_triangle_fan(vertices, uvs, colors)


    
    
