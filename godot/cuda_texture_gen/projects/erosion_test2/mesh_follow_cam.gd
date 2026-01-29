@tool
extends Node3D

@export var follow_camera := true
@export var snap_size := 64.0  # Size of grid squares to snap to
@export var water_height := 0.0  # Y position to lock to
@export var base_scale := 1.0  # Base scale at height 0
@export var scale_step_height := 64.0  # Height needed to increase scale by one step
@export var scale_step_amount := 2.0  # How much to scale each step (2.0 = double)

var target_position := Vector3.ZERO

func _process(_delta):
    if not follow_camera:
        return
    
    var camera: Camera3D = null
    
    # In editor, get editor camera
    if Engine.is_editor_hint():
        var editor_interface = EditorInterface if EditorInterface else null
        if editor_interface:
            var editor_viewport = EditorInterface.get_editor_viewport_3d(0)
            if editor_viewport:
                camera = editor_viewport.get_camera_3d()
    else:
        # In game, get active camera
        camera = get_viewport().get_camera_3d()
    
    if camera:
        # Get camera position
        var cam_pos = camera.global_position
        
        # Snap to grid
        var snapped_x = round(cam_pos.x / snap_size) * snap_size
        var snapped_z = round(cam_pos.z / snap_size) * snap_size
        
        target_position = Vector3(snapped_x, water_height, snapped_z)
        global_position = target_position
        
        # Scale based on camera height in discrete steps
        var height_above_water = max(0, cam_pos.y - water_height)
        var scale_steps = floor(height_above_water / scale_step_height)
        var new_scale = base_scale * pow(scale_step_amount, scale_steps)
        scale = Vector3(new_scale, 1.0, new_scale)
