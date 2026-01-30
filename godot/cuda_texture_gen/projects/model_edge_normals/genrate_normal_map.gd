"""
"""
@tool
extends Node





@export var image: Image

@export var texture: Texture2D


@export var trigger_update := false:
    set(value):
        if value:
            do_something()
            trigger_update = false  # Reset immediately
           


func do_something():
    print("do_something()...")
    
    image = NormalMapUtils.generate_bevel_normal_map(256, 256, 64, 8, 8, 16)
    
    texture = ImageTexture.create_from_image(image)
    

    
    
    
    
