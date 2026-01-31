"""

tools to deal with geometry

typically geometry is stored as Array[Dictionary]

where the array's dictionaries represent quads or ngons
they have "verts" and "uvs" (PackedVector3Array and PackedVector2Array)

"""
#@tool
class_name GeometryUtils

#region HELPER

## get normal direction based on first three verts for cube project
static func get_face_normal(vertices: PackedVector3Array) -> Vector3:
    assert(vertices.size() >= 3)
    var v1 := vertices[1] - vertices[0]
    var v2 := vertices[2] - vertices[0]
    return v1.cross(v2).normalized()


## apply Vector3, Basis, or Transform3D to verts (in place)
static func transform_verts(verts: PackedVector3Array, op) -> void:
    # Vector3 → offset
    if op is Vector3:
        var offset: Vector3 = op
        for i in verts.size():
            verts[i] = verts[i] + offset
        return

    # Basis → linear transform
    if op is Basis:
        var basis: Basis = op
        for i in verts.size():
            verts[i] = basis * verts[i] # note correct way
        return

    # Transform3D → full transform
    if op is Transform3D:
        var xf: Transform3D = op
        for i in verts.size():
            verts[i] = xf * verts[i] # note correct way
        return

    push_error("transform_verts: unsupported type '%s'" % [typeof(op)])

## cycle the verts around in the array, changes the edge order
static func cycle_verts(verts: PackedVector3Array, steps: int) -> PackedVector3Array:
    
    var count := verts.size()
    if count == 0:
        return verts

    steps = steps % count
    if steps < 0:
        steps += count

    var result := PackedVector3Array()
    result.resize(count)

    for i in count:
        result[i] = verts[(i + steps) % count]

    return result

#endregion

#region QUAD_OPS

## split a quad into two quads
static func split_quad(
    verts: PackedVector3Array,
    uvs: PackedVector2Array,
    edge_index: int,
    offset: float
) -> Array[Dictionary]:
    assert(verts.size() == 4)
    assert(uvs.size() == 4)
    
    var i0 := edge_index % 4
    var i1 := (edge_index + 1) % 4
    var i2 := (edge_index + 2) % 4
    var i3 := (edge_index + 3) % 4
    
    # Split point on the selected edge (i0 -> i1)
    var v_split_near := verts[i0].lerp(verts[i1], offset)
    var uv_split_near := uvs[i0].lerp(uvs[i1], offset)
    
    # Split point on the opposite edge (i3 -> i2)
    # Note: lerping in same direction to maintain offset meaning
    var v_split_far := verts[i3].lerp(verts[i2], offset)
    var uv_split_far := uvs[i3].lerp(uvs[i2], offset)
    
    # Quad A: i0 → split_near → split_far → i3
    var quad_a_verts = PackedVector3Array([
        verts[i0],
        v_split_near,
        v_split_far,
        verts[i3]
    ])
    var quad_a_uvs = PackedVector2Array([
        uvs[i0],
        uv_split_near,
        uv_split_far,
        uvs[i3]
    ])
    
    # Quad B: split_near → i1 → i2 → split_far
    var quad_b_verts = PackedVector3Array([
        v_split_near,
        verts[i1],
        verts[i2],
        v_split_far
    ])
    var quad_b_uvs = PackedVector2Array([
        uv_split_near,
        uvs[i1],
        uvs[i2],
        uv_split_far
    ])
    
    return [
        {"verts": quad_a_verts, "uvs": quad_a_uvs},
        {"verts": quad_b_verts, "uvs": quad_b_uvs}
    ]
    
## split a quad, then extrude the edge, used to create a bevel like shape
static func split_and_extrude(
    verts: PackedVector3Array,
    uvs: PackedVector2Array,
    edge_index: int,
    split_offset: float,      # Renamed: clearer than just "offset"
    extrude_amount: float,      # Renamed: clearer than "bevel_amount"
    extrude_first_quad: bool = false  # Which side to extrude
    ) -> Array[Dictionary]:
    
    var normal := get_face_normal(verts)
    
    var split = GeometryUtils.split_quad(
        verts,
        uvs,
        (edge_index + 1) % 4,
        split_offset)
    
    var quad1 = split[0]
    var quad2 = split[1]
    
    if extrude_first_quad:
        quad2.verts[1] += normal * extrude_amount
        quad2.verts[2] += normal * extrude_amount
    else:
        quad1.verts[3] += normal * extrude_amount
        quad1.verts[0] += normal * extrude_amount
        
    return split

#endregion


#region UVS

## determine dominant axis of a normal for cube project
static func get_major_axis(input: Vector3) -> int:
    
    input.x = abs(input.x) 
    input.y = abs(input.y)
    input.z = abs(input.z)
    
    if input.x >= input.y:
        if input.x >= input.z: return Vector3.AXIS_X
        else: return Vector3.AXIS_Z
    else:
        if input.y >= input.z: return Vector3.AXIS_Y
        else: return Vector3.AXIS_Z



static func project_vertex_to_uv(vert: Vector3, axis: int) -> Vector2:
    
    var uv = Vector2.ZERO
    match axis:
        Vector3.AXIS_X: uv = Vector2(-vert.z, -vert.y)
        Vector3.AXIS_Y: uv = Vector2(vert.x, vert.z)
        Vector3.AXIS_Z: uv = Vector2(vert.x, -vert.y)    
    return uv

## get cube projected uv positons from an ngon's vertices
static func get_cube_projected_uvs(
    vertices: PackedVector3Array,
    correct_reversed_uvs: bool = false,
    uv_scale: Vector2 = Vector2(1,1),
    uv_offset: Vector2 = Vector2(1,1)
    ) -> PackedVector2Array:
    
    assert(vertices.size() >= 3)
    
    var uvs := PackedVector2Array() # build uv array
    uvs.resize(vertices.size()) # set it's size
    
    var normal := get_face_normal(vertices)
    var axis := get_major_axis(normal)
    
    for i in vertices.size():
        var uv := project_vertex_to_uv(vertices[i], axis)            
        if correct_reversed_uvs and normal[axis] > 0.0: uv.x = -uv.x # reversed uv correction
        uvs[i] = uv * uv_scale + uv_offset
        
    return uvs

#endregion



#region BUILD

## extrude an ngon to make a solid shape
static func extrude_ngon(
        verts: PackedVector3Array,
        uvs: PackedVector2Array,
        extrude_length: float,
        optional_normal: Vector3 = Vector3.ZERO, # optional normal (if zero, compute normal)
        front_face: bool = true, # keep front face (orginal)
        back_face: bool = true, # make back face
        side_quads: bool = true,
    ) -> Array[Dictionary]:

    assert(verts.size() >= 3) # at least triangle
    if uvs.is_empty(): uvs.resize(verts.size()) # allow no uv (empty)
    assert(verts.size() == uvs.size()) # verts and uvs must be same size

    var out: Array[Dictionary] = []
        
    if not optional_normal: 
        optional_normal = get_face_normal(verts).normalized()
        
    var offset: Vector3 = optional_normal * extrude_length

    var count := verts.size()

    if front_face:
        out.append({
            "verts": verts,
            "uvs": uvs
        })

    # make back face with reversed winding
    if back_face:
        var back_verts := PackedVector3Array()
        var back_uvs := PackedVector2Array()

        for i in range(count):
            back_verts.append(verts[i] + offset)
            back_uvs.append(uvs[i])  # reuse UVs (not perfect, but acceptable)

        # reverse winding so normals face outward
        back_verts.reverse()
        back_uvs.reverse()

        out.append({
            "verts": back_verts,
            "uvs": back_uvs
        })

    if side_quads:
        for i in range(count):
            var i2 := (i + 1) % count

            var a := verts[i]
            var b := verts[i2]
            var c := b + offset
            var d := a + offset

            var ua := uvs[i]
            var ub := uvs[i2]
            
            var uc := ub
            var ud := ua
            
            var correct_extrude_uv := true
            if correct_extrude_uv:
                var correction := -extrude_length
                if ua.x == ub.x:
                    uc.x += correction
                    ud.x += correction
                if ua.y == ub.y:
                    uc.y += correction
                    ud.y += correction
                
            var side_uvs := PackedVector2Array([ua, ub, uc, ud])
            
            out.append({
                #"verts": PackedVector3Array([a, b, c, d]),
                "verts": PackedVector3Array([d, c, b, a]), # reverse winding
                "uvs": side_uvs
            })

    return out


## works to bend a pipe, but need to set the pipe length to PI / 2.0 to get it to bend on the grid
## currently only bends a vertical pipe to face x+
static func simple_deform_bend(
    point: Vector3,
    bend_angle: float,  # Total bend angle in degrees (e.g., 90)
    pipe_length: float = 1.0  # The actual length of your pipe
) -> Vector3:
    
    if bend_angle == 0.0: return point
    
    var angle_rad := deg_to_rad(bend_angle)
    var radius := pipe_length / angle_rad  # Radius based on actual pipe length
    
    # Normalize to 0-1 based on actual pipe length
    var t := point.y / pipe_length
    
    # Current angle at this point
    var theta := angle_rad * t
    
    # Account for X offset from centerline (for pipe thickness)
    var r := radius - point.x
    
    # Calculate position on circular arc
    var new_x := r * sin(theta)
    var new_y := radius - r * cos(theta)
    
    return Vector3(new_y, new_x, point.z)
    
    
static func simple_deform_bend_points(points, bend_angle: float, pipe_length: float) -> void:
    for i in points.size():
        points[i] = simple_deform_bend(points[i], 90, pipe_length)


static func get_default_verts() -> PackedVector3Array:
    return PackedVector3Array([
        Vector3(0,0,0),
        Vector3(1,0,0),
        Vector3(1,0,1),
        Vector3(0,0,1),
    ])

static func get_default_uvs() -> PackedVector2Array:
    return PackedVector2Array([
    Vector2(0,0),
    Vector2(1,0),
    Vector2(1,1),
    Vector2(0,1),
])


## get a cannon cube by using extrude
static func get_cube() -> Array[Dictionary]:
    
    #var verts := get_default_verts()
    #transform_verts(verts, Vector3(0, 1, 0)) # top face
    #var uvs := PackedVector2Array() # empty uvs (will be generated)

    var verts := PackedVector3Array([Vector3(1,1,0), Vector3(0,1,0), Vector3(0,0,0), Vector3(1,0,0)])
    var uvs := PackedVector2Array([Vector2(0,0), Vector2(1,0), Vector2(1,1), Vector2(0,1)])
    
    var result: Array[Dictionary] = []
    for quad in extrude_ngon(verts, uvs, 1.0):
        #quad.uvs = get_cube_projected_uvs(quad.verts)
        result.append(quad)
    return result
    




## get a tube as quads
static func get_tube_quads(
        radius: float,
        height: float,
        radial_segments: int,
        height_segments: int,
        reverse: bool = false,
        uv_scale: Vector2 = Vector2(1, 1)
    ) -> Array:

    var quads: Array = []
    var stride: int = radial_segments + 1

    var verts: Array = []
    var uvs: Array = []

    # Build vertex + uv grid
    for y in range(height_segments + 1):
        var v: float = float(y) / float(height_segments)
        var py: float = v * height   # floor → top



        for x in range(radial_segments + 1):
            var u: float = float(x) / float(radial_segments)
            var angle: float = u * TAU

            var px: float = cos(angle) * radius
            var pz: float = sin(angle) * radius

            uvs.append(Vector2(-u * uv_scale.x, -v * uv_scale.y))
            verts.append(Vector3(px, py, pz))

    # Build quads
    for y in range(height_segments):
        for x in range(radial_segments):

            var a: int = y * stride + x
            var b: int = a + stride
            var c: int = a + 1
            var d: int = b + 1

            if not reverse:
                quads.append({
                    "verts": PackedVector3Array([verts[a], verts[c], verts[d], verts[b]]),
                    "uvs": PackedVector2Array([uvs[a], uvs[c], uvs[d], uvs[b]])
                })
            else:
                quads.append({
                    "verts": PackedVector3Array([verts[a], verts[b], verts[d], verts[c]]),
                    "uvs": PackedVector2Array([uvs[a], uvs[b], uvs[d], uvs[c]])
                })

    return quads



#endregion

    
