"""


"""
#@tool
class_name GeometryUtils

## get normal direction based on first three verts for cube project
static func get_normal(vertices: PackedVector3Array) -> Vector3:
    assert(vertices.size() >= 3)
    var v1 := vertices[1] - vertices[0]
    var v2 := vertices[2] - vertices[0]
    return v1.cross(v2).normalized()

## reverse the order of a PackedVector3Array
static func reverse_verts(verts: PackedVector3Array) -> PackedVector3Array:
    var result := PackedVector3Array()
    var count := verts.size()
    result.resize(count)
    for i in count:
        result[count - 1 - i] = verts[i]
    return result

## reverse the order of a PackedVector2Array
static func reverse_uvs(uvs: PackedVector2Array) -> PackedVector2Array:
    var result := PackedVector2Array()
    var count := uvs.size()
    result.resize(count)
    for i in count:
        result[count - 1 - i] = uvs[i]
    return result

## apply Basis to PackedVector3Array
static func rotate_verts(verts: PackedVector3Array, basis: Basis) -> PackedVector3Array:
    var result := PackedVector3Array()
    result.resize(verts.size())
    for i in verts.size():
        result[i] *= basis
    return result

## apply Transform3D to PackedVector3Array
static func transform_verts(verts: PackedVector3Array, transform: Transform3D) -> PackedVector3Array:
    var result := PackedVector3Array()
    result.resize(verts.size())
    for i in verts.size():
        result[i] *= transform
    return result

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
    
    var normal := get_normal(verts)
    
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


#region BUILD


    




#endregion

    
