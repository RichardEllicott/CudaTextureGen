"""

normal map image generation and manipulation


example:

var image = NormalMapUtils.generate_bevel_normal_map(256, 256, 64, 8, 8, 16)
var texture = ImageTexture.create_from_image(image)

"""
#@tool
class_name NormalMapUtils

## create an image with beveled edges
static func generate_bevel_normal_map(
        width: int,
        height: int,
        bevel_top: float,
        bevel_right: float,
        bevel_bottom: float,
        bevel_left: float
    ) -> Image:

    var img: Image = Image.create(width, height, false, Image.FORMAT_RGB8)

    for y in height:
        for x in width:

            var nx: float = 0.0
            var ny: float = 0.0

            # LEFT EDGE
            if bevel_left != 0.0:
                var bw_l: float = abs(bevel_left)
                var dist_l: float = float(x) / bw_l
                var t_l: float = clamp(1.0 - dist_l, 0.0, 1.0)
                var b_l: float = t_l * t_l * (3.0 - 2.0 * t_l)
                nx -= b_l * sign(bevel_left)

            # RIGHT EDGE
            if bevel_right != 0.0:
                var bw_r: float = abs(bevel_right)
                var dist_r: float = float(width - 1 - x) / bw_r
                var t_r: float = clamp(1.0 - dist_r, 0.0, 1.0)
                var b_r: float = t_r * t_r * (3.0 - 2.0 * t_r)
                nx += b_r * sign(bevel_right)

            # TOP EDGE
            if bevel_top != 0.0:
                var bw_t: float = abs(bevel_top)
                var dist_t: float = float(y) / bw_t
                var t_t: float = clamp(1.0 - dist_t, 0.0, 1.0)
                var b_t: float = t_t * t_t * (3.0 - 2.0 * t_t)
                ny -= b_t * sign(bevel_top)

            # BOTTOM EDGE
            if bevel_bottom != 0.0:
                var bw_b: float = abs(bevel_bottom)
                var dist_b: float = float(height - 1 - y) / bw_b
                var t_b: float = clamp(1.0 - dist_b, 0.0, 1.0)
                var b_b: float = t_b * t_b * (3.0 - 2.0 * t_b)
                ny += b_b * sign(bevel_bottom)

            # Compute Z from X/Y
            var nz: float = sqrt(max(0.0, 1.0 - nx * nx - ny * ny))

            # Encode tangent-space normal
            var r: float = nx * 0.5 + 0.5
            var g: float = ny * 0.5 + 0.5
            var bcol: float = nz * 0.5 + 0.5

            img.set_pixel(x, y, Color(r, g, bcol))

    return img

## combine two normal maps
static func combine_normal_maps(
        img_a: Image,
        img_b: Image
    ) -> Image:

    assert(img_a.get_width() == img_b.get_width())
    assert(img_a.get_height() == img_b.get_height())

    var width: int = img_a.get_width()
    var height: int = img_a.get_height()

    var out_img: Image = Image.create(width, height, false, Image.FORMAT_RGB8)

    for y in height:
        for x in width:

            var ca: Color = img_a.get_pixel(x, y)
            var cb: Color = img_b.get_pixel(x, y)

            # Decode normals from RGB → [-1,1]
            var ax: float = ca.r * 2.0 - 1.0
            var ay: float = ca.g * 2.0 - 1.0
            var az: float = ca.b * 2.0 - 1.0

            var bx: float = cb.r * 2.0 - 1.0
            var by: float = cb.g * 2.0 - 1.0
            var bz: float = cb.b * 2.0 - 1.0

            # Combine (add) normals
            var nx: float = ax + bx
            var ny: float = ay + by
            var nz: float = az + bz

            # Normalize
            var len: float = sqrt(nx * nx + ny * ny + nz * nz)
            if len > 0.0:
                nx /= len
                ny /= len
                nz /= len

            # Encode back to RGB
            var r: float = nx * 0.5 + 0.5
            var g: float = ny * 0.5 + 0.5
            var b: float = nz * 0.5 + 0.5

            out_img.set_pixel(x, y, Color(r, g, b))

    return out_img
