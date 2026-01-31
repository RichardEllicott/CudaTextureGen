"""

normal map image generation and manipulation


example:

var image = NormalMapUtils.generate_bevel_normal_map(256, 256, 64, 8, 8, 16)
var texture = ImageTexture.create_from_image(image)

"""
#@tool
class_name ImageUtils

## simple blur
static func box_blur(
        src: Image,
        radius: int
    ) -> Image:

    var width: int = src.get_width()
    var height: int = src.get_height()

    var dst: Image = Image.create(width, height, false, src.get_format())

    var size: int = radius * 2 + 1
    var area: float = float(size * size)

    for y in height:
        for x in width:

            var r: float = 0.0
            var g: float = 0.0
            var b: float = 0.0
            var a: float = 0.0

            # accumulate kernel
            for ky in range(-radius, radius + 1):
                var sy: int = clamp(y + ky, 0, height - 1)

                for kx in range(-radius, radius + 1):
                    var sx: int = clamp(x + kx, 0, width - 1)

                    var c: Color = src.get_pixel(sx, sy)
                    r += c.r
                    g += c.g
                    b += c.b
                    a += c.a

            # average
            r /= area
            g /= area
            b /= area
            a /= area

            dst.set_pixel(x, y, Color(r, g, b, a))

    return dst

## draw a circle
static func draw_circle(
        img: Image,
        center: Vector2i,
        radius: int,
        color: Color,
        thickness: int = -1   # -1 = filled circle
    ) -> void:
        
        
    

    var width: int = img.get_width()
    var height: int = img.get_height()

    var r2: int = radius * radius
    var inner_r2: int = (radius - thickness) * (radius - thickness)

    for y in range(center.y - radius, center.y + radius + 1):
        if y < 0 or y >= height:
            continue

        for x in range(center.x - radius, center.x + radius + 1):
            if x < 0 or x >= width:
                continue

            var dx: int = x - center.x
            var dy: int = y - center.y
            var d2: int = dx * dx + dy * dy

            # Filled circle
            if thickness < 0:
                if d2 <= r2:
                    img.set_pixel(x, y, color)
                continue

            # Outline circle
            if d2 <= r2 and d2 >= inner_r2:
                img.set_pixel(x, y, color)


static func draw_rectangle(
        img: Image,
        rect: Rect2i,
        color: Color,
        thickness: int = -1   # -1 = filled
    ) -> void:

    var width: int = img.get_width()
    var height: int = img.get_height()

    var x0: int = max(0, rect.position.x)
    var y0: int = max(0, rect.position.y)
    var x1: int = min(width - 1, rect.position.x + rect.size.x - 1)
    var y1: int = min(height - 1, rect.position.y + rect.size.y - 1)

    # Filled rectangle
    if thickness < 0:
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                img.set_pixel(px, py, color)
        return

    # Outline rectangle
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):

            var on_left: bool = px < x0 + thickness
            var on_right: bool = px > x1 - thickness
            var on_top: bool = py < y0 + thickness
            var on_bottom: bool = py > y1 - thickness

            if on_left or on_right or on_top or on_bottom:
                img.set_pixel(px, py, color)
