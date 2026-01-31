"""
"""
@tool
extends MeshInstance3D





@export var image2: Image

@export var texture2: Texture2D


@export var trigger_update := false:
    set(value):
        if value:
            do_something()
            trigger_update = false  # Reset immediately
           

func test_bevel_gen():
    image2 = NormalMapUtils.generate_bevel_normal_map(256, 256, 64, 8, 8, 16)
    texture2 = ImageTexture.create_from_image(image2)

func test_image_draw_to_normal():
    
    var image_size := Vector2i(256, 256)
    image2 = Image.create_empty(image_size.x, image_size.y, false, Image.FORMAT_RGBA8)
    image2.fill(Color.WHITE)
    
    ImageUtils.draw_circle_on_image(image2, image_size.x / 2, image_size.y / 2, 32, Color.BLACK)
    image2 = ImageUtils.blur_image_box(image2, 4.0)
    image2 = NormalMapUtils.heightmap_to_normal_map(image2, 1.0)
    
    
    

func do_something():
    print("do_something()...")
    
    #test_bevel_gen()
    test_image_draw_to_normal()
    
    
    
    

    
    
    
    
