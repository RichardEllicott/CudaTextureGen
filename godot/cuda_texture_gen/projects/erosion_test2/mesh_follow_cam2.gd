@tool
extends Node3D

enum FollowMode {
    CAMERA_POSITION,  # Follow camera XZ position
    CAMERA_LOOK_AT    # Follow where camera is looking at ground plane
}

@export var follow_camera := true
@export var follow_mode := FollowMode.CAMERA_POSITION
@export var snap_size := 1.0  # Size of grid squares to snap to
@export var water_height := 0.0  # Y position to lock to
@export var max_look_distance := 100.0  # Max distance for look-at mode
@export var base_scale := 1.0  # Base scale at height 0
@export var scale_step_height := 10.0  # Height needed to increase scale by one step
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
        var cam_pos = camera.global_position
        var target_xz := Vector2.ZERO
        
        if follow_mode == FollowMode.CAMERA_POSITION:
            # Simple mode - follow camera XZ position
            target_xz = Vector2(cam_pos.x, cam_pos.z)
        else:
            # Look-at mode - raycast to ground plane
            var cam_forward = -camera.global_transform.basis.z
            
            # Check if looking above or below horizon
            if cam_forward.y < 0:
                # Looking down - raycast to ground plane at water_height
                var ray_origin = cam_pos
                var plane_normal = Vector3.UP
                var plane_point = Vector3(0, water_height, 0)
                
                # Ray-plane intersection
                var denom = plane_normal.dot(cam_forward)
                if abs(denom) > 0.0001:
                    var t = (plane_point - ray_origin).dot(plane_normal) / denom
                    if t >= 0:
                        var intersection = ray_origin + cam_forward * t
                        var distance = cam_pos.distance_to(intersection)
                        
                        # Clamp to max distance
                        if distance > max_look_distance:
                            intersection = cam_pos + cam_forward.normalized() * max_look_distance
                            intersection.y = water_height
                        
                        target_xz = Vector2(intersection.x, intersection.z)
                    else:
                        # Fallback to camera position
                        target_xz = Vector2(cam_pos.x, cam_pos.z)
                else:
                    # Ray parallel to plane - fallback
                    target_xz = Vector2(cam_pos.x, cam_pos.z)
            else:
                # Looking up/at horizon - use inverted forward direction
                var inverted_forward = Vector3(cam_forward.x, -cam_forward.y, cam_forward.z)
                
                var ray_origin = cam_pos
                var plane_normal = Vector3.UP
                var plane_point = Vector3(0, water_height, 0)
                
                var denom = plane_normal.dot(inverted_forward)
                if abs(denom) > 0.0001:
                    var t = (plane_point - ray_origin).dot(plane_normal) / denom
                    if t >= 0:
                        var intersection = ray_origin + inverted_forward * t
                        var distance = cam_pos.distance_to(intersection)
                        
                        # Clamp to max distance
                        if distance > max_look_distance:
                            intersection = cam_pos + inverted_forward.normalized() * max_look_distance
                            intersection.y = water_height
                        
                        target_xz = Vector2(intersection.x, intersection.z)
                    else:
                        target_xz = Vector2(cam_pos.x, cam_pos.z)
                else:
                    target_xz = Vector2(cam_pos.x, cam_pos.z)
        
        # Snap to grid
        var snapped_x = round(target_xz.x / snap_size) * snap_size
        var snapped_z = round(target_xz.y / snap_size) * snap_size
        
        target_position = Vector3(snapped_x, water_height, snapped_z)
        global_position = target_position
        
        # Scale based on camera height in discrete steps
        var height_above_water = max(0, cam_pos.y - water_height)
        var scale_steps = floor(height_above_water / scale_step_height)
        var new_scale = base_scale * pow(scale_step_amount, scale_steps)
        scale = Vector3(new_scale, 1.0, new_scale)
