"""



"""
@tool
extends Node
class_name PythonSocketClient

#region Vars
#endregion

@export var enabled := true:
    get: return enabled
    set(value):
        enabled = value
        if enabled: _start()
        else: _stop()

@export var port := 8765

var _websocket: WebSocketPeer
var _timer := 0.0
var _connected := false
var _reconnecting := false
var _last_header
var _last_log_msg


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

func _process(delta: float) -> void:
    if not enabled:
        return

    _timer += delta

    if _websocket:
        _websocket.poll()

    match _websocket.get_ready_state():
        WebSocketPeer.STATE_CONNECTING:
            pass

        WebSocketPeer.STATE_OPEN:
            if not _connected:
                _connected = true
                log_msg("Connected to Python daemon")
                _on_connect()
                
            _on_data()

        WebSocketPeer.STATE_CLOSING:
            log_msg("Closing...")

        WebSocketPeer.STATE_CLOSED:
            if _connected:
                _connected = false
                log_msg("Disconnected from daemon")

            if not _reconnecting:
                _reconnecting = true
                _attempt_reconnect()

# ================================================================
# [Start / Stop]
# ----------------------------------------------------------------
func _start():
    _connected = false
    _reconnecting = false

    _websocket = WebSocketPeer.new()

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
        var pkt = _websocket.get_packet()

        if _websocket.was_string_packet():
            var header = JSON.parse_string(pkt.get_string_from_utf8())
            log_msg("Header: %s" % header)
            _last_header = header

        else:
            var bytes = pkt
            log_msg("Binary payload: %s bytes" % bytes.size())

            if _last_header and _last_header.get("type") == "image":
                _handle_image(bytes, _last_header)
                pass


func _handle_image(bytes: PackedByteArray, header: Dictionary) -> void:
    print("try to _handle_image...")
        
    # Validate header
    var width: int = header.get("width", 0)
    var height: int = header.get("height", 0)
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
    await get_tree().create_timer(1.0).timeout
    log_msg("Reconnecting...")

    _websocket = WebSocketPeer.new()
    _websocket.connect_to_url("ws://localhost:%d" % port)

    _reconnecting = false

# ================================================================
# [Commands]
# ----------------------------------------------------------------


# override in subclasses
func _on_connect() -> void:
    #cmd_ping()
    cmd_send_array_test()



func send_command(msg: Dictionary):
    var err = _websocket.send_text(JSON.stringify(msg))
    if err != OK:
        log_msg("Send failed: %s" % err)


func cmd_ping():
    var msg = {"cmd": "ping"}
    send_command(msg)


func cmd_send_array_test():
    var msg = {"cmd": "send_array_test"}
    send_command(msg)

    
    
    
    
    
