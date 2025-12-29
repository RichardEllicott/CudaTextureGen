"""



"""
@tool
extends Node
class_name PythonSocketClient

@export var enabled := true:
    get: return enabled
    set(value):
        enabled = value
        if enabled: _start()
        else: _stop()

@export var port := 8765 # connection port
@export var max_buffer_bytes := 1024 * 1024 * 4 # max binary for one chunk, if we exceed this we get a dissconnection

var _websocket: WebSocketPeer # connection
var _timer := 0.0 # seconds elapsed
var _connected := false # is connected
var _reconnecting := false # is reconnecting
var _last_header # last header (from json string)

var _last_log_msg # last logged message

var _binary_buffer := PackedByteArray() # buffer for incoming binary packets

signal connected()

@export var connection_state = -1


@export_group("Daemon")

@export var enable_daemon: bool = false:
    get:
        return enable_daemon
    set(value):
        enable_daemon = value
        if not enable_daemon:
            kill_python_daemon()
        
#@export var script_path := "C:/Users/Richard/github/cuda_texture_gen/lib/python_daemon.py"
@export var script_path := ProjectSettings.globalize_path("res://../../lib/python_daemon.py")


@export_group("")




# ================================================================
# [Debug]
# ----------------------------------------------------------------
func log_msg(msg):
    _last_log_msg = msg
    print("💻 ", msg)

# ================================================================
# [Lifecycle]
# ----------------------------------------------------------------
func _ready():
    if enabled:
        _start()




#region UNATTACHED_PYTHON_LAUNCH
var _python_daemon_pid := 0
func launch_python_daemon():
    
    if _python_daemon_pid == -1: # if pid is -1
        _python_daemon_pid = 0 #  set to 0
    
    if _python_daemon_pid: return            
    _python_daemon_pid = OS.create_process("python", [script_path])

func kill_python_daemon():
    OS.kill(_python_daemon_pid)
    _python_daemon_pid = 0
    
#endregion

#region THREADED_PYTHON_LAUNCH
#func _python_daemon_thread_func():
    ##return OS.execute("python", [script_path], [], false, true)
    #return OS.execute("python", [script_path], []) # no console
#
#var _python_daemon_thread: Thread = null
#
#func launch_python_daemon_thread():
    #
    #print("launch_python_daemon...")
    #
    ## If we have a thread and it's finished, clean it up
    #if _python_daemon_thread and not _python_daemon_thread.is_alive():
        #var exit_code = _python_daemon_thread.wait_to_finish()
        #print("Daemon exited with:", exit_code)
        #_python_daemon_thread = null  # important
#
    ## If no thread exists or it's finished, create a new one
    #if _python_daemon_thread == null:
        #print("start thread")
        #_python_daemon_thread = Thread.new()
        #_python_daemon_thread.start(_python_daemon_thread_func)
#endregion


#
#func _daemon_thread_func(userdata):
    ## This blocks, but only inside the thread — main thread stays free
    #OS.execute("python", ["path/to/daemon.py"], [], false, true)
    #return null
#
#
#func launch_daemon():
    #if _daemon_thread == null:
        #_daemon_thread = Thread.new()
        #_daemon_thread.start(self, "_daemon_thread_func")

func _exit_tree():
    print("🐱 _exit_tree()...")
    #if _python_daemon_thread:
        #_python_daemon_thread.cancel_free()
        #_python_daemon_thread.wait_to_finish()
        
    kill_python_daemon()
        
        


    



func _process(delta: float) -> void:
    if not enabled:
        return

    _timer += delta

    if _websocket:
        _websocket.poll()
        
        
    connection_state = _websocket.get_ready_state()

    match connection_state:
        WebSocketPeer.STATE_CONNECTING: # 0
            log_msg("Connecting for %.2f seconds" % _timer)
            if _timer > 1.0:
                if enable_daemon:
                    launch_python_daemon()
                
        WebSocketPeer.STATE_OPEN: # 1
            if not _connected:
                _connected = true
                log_msg("Connected to Python daemon")
                _on_connect()
                connected.emit()
                
            _on_data()

        WebSocketPeer.STATE_CLOSING: # 2
            log_msg("Closing...")

        WebSocketPeer.STATE_CLOSED: # 3
            if _connected:
                _connected = false
                log_msg("Disconnected from daemon")

            if not _reconnecting:
                _attempt_reconnect()

# ================================================================
# [Start / Stop]
# ----------------------------------------------------------------
func _start():
    _connected = false
    _reconnecting = false
    
    _timer = 0.0

    _websocket = WebSocketPeer.new()
    
    _websocket.set_inbound_buffer_size(max_buffer_bytes)
    _websocket.set_outbound_buffer_size(max_buffer_bytes)

    var url := "ws://localhost:%d" % port
    log_msg("client connect to: %s" % url)

    var err := _websocket.connect_to_url(url)
    if err != OK:
        log_msg("Connect error: %s" % err)

func _stop():
    if _websocket:
        _websocket.close()

    _connected = false
    _reconnecting = false

    log_msg("WebSocket disabled")

# ================================================================
# [Data Handling]
# ----------------------------------------------------------------
func _on_data():
    while _websocket.get_available_packet_count() > 0:
        
        var pkt := _websocket.get_packet() # get packet (bytes)

        if _websocket.was_string_packet(): # if a string packet            
            _last_header = JSON.parse_string(pkt.get_string_from_utf8()) # parse as a string
            _binary_buffer.clear() # ensure binary buffer is clear (as header may indicate new binary)
        else:
            var size_bytes = _last_header.get("size_bytes", 0)
            var type = _last_header.get("type", null)
            
            _binary_buffer += pkt # Append binary chunk

            # If we have enough bytes, process the buffer
            if _binary_buffer.size() >= size_bytes:
                
                if type == "image":
                    _handle_image(_binary_buffer, _last_header)
                _last_header = null
                _binary_buffer.clear()




func _handle_image(bytes: PackedByteArray, header: Dictionary) -> void:
        
    # Validate header
    var width: int = int(header.get("width", 0))
    var height: int = int(header.get("height", 0))
    var format: String = header.get("format", "")
    #var label: String = header.get("label", "image")

    if width <= 0 or height <= 0:
        push_error("Invalid image dimensions")
        return
    
    var img: Image
    if format == "f32":
        
        img = Image.create_from_data(width, height, false, Image.FORMAT_RF, bytes) # will look red
        #img.convert(Image.FORMAT_RGBA8) # will still look red (but keeps floats)
        img.convert(Image.FORMAT_L8) # conert to b&w (but 8 bit)
        
    elif format == "rgba8":
        img = Image.create_from_data(width, height, false, Image.FORMAT_RGBA8, bytes)
    else:
        push_error("Unknown image format: %s" % format)
        return

    # Turn it into a texture
    var tex := ImageTexture.new()
    tex.set_image(img)

    # Find a TextureRect child (or create one)
    var rect := $TextureRect if has_node("TextureRect") else null
    if rect:
        rect.texture = tex
    else:
        print("No TextureRect child found")




# ================================================================
# [Reconnect Logic]
# ----------------------------------------------------------------
func _attempt_reconnect():
    
    if _reconnecting: return # skip if already reconnecting
    
    _reconnecting = true
    
    #await get_tree().create_timer(1.0).timeout # ❓ short delay
    log_msg("Reconnecting...")

    _websocket = WebSocketPeer.new()
    _websocket.connect_to_url("ws://localhost:%d" % port)

    _reconnecting = false

# ================================================================
# [Commands]
# ----------------------------------------------------------------


# override in subclasses
func _on_connect() -> void:
    
    
    
    _last_header = null
    
    #await get_tree().create_timer(1.0).timeout # ❓maybe not required
    
    cmd_send_array_test()
    #cmd_ping()
    #cmd_ping_binary()



func send_command(msg: Dictionary):
    var err = _websocket.send_text(JSON.stringify(msg))
    if err != OK:
        log_msg("Send failed: %s" % err)


func cmd_ping():
    var msg = {"cmd": "ping"}
    send_command(msg)


func cmd_ping_binary():
    var msg = {"cmd": "ping_binary"}
    send_command(msg)
    
    
func cmd_send_array_test():
    var msg = {
        "cmd": "send_array_test",
        "width": 32,
        "height": 32,
        "misc": 123,
        }
    
    send_command(msg)

    
    
    
    
    
