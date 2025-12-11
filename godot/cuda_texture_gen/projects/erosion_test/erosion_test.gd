"""
"""
@tool
extends Node3D


func _ready() -> void:
    
    var i := 0
        
    var set_size := Vector2(1,1) * 512
    
    
    for child in get_children():
        
        if child is TextureRect:
            var child2: TextureRect = child
            
            child2.size = set_size
            
            i += 1
    
    pass
