"""




"""
@tool
extends TextureRect

@export var enabled := true

@export var speed_scale = 1.0


@export var animation_folder_path: String = "res://textures/animation_test"

func create_animated_texture_from_folder(folder_path: String) -> AnimatedTexture:
    var animated_texture := AnimatedTexture.new()
    #animated_texture.set_fps(fps)  # Use setter in Godot 4

    var dir := DirAccess.open(folder_path)
    if dir == null:
        push_error("Failed to open folder: %s" % folder_path)
        return null

    var texture_paths := []
    dir.list_dir_begin()
    while true:
        var file_name = dir.get_next()
        if file_name == "":
            break
        if not dir.current_is_dir() and file_name.ends_with(".png"):
            texture_paths.append(folder_path + "/" + file_name)
    dir.list_dir_end()

    texture_paths.sort()

    print(texture_paths)
    
    animated_texture.frames = texture_paths.size()
    
    for i in texture_paths.size():
        var path = texture_paths[i]
        var _texture = load(path)
        #if texture is Texture2D:
        animated_texture.set_frame_texture(i, _texture)
        


    return animated_texture


func _ready():
    
    if not enabled:
        return
        
    var ani := create_animated_texture_from_folder(animation_folder_path)
    ani.speed_scale = speed_scale
    texture = ani
    
    var save_path = animation_folder_path + "/generated_anim.tres"
    ResourceSaver.save(ani, save_path)
