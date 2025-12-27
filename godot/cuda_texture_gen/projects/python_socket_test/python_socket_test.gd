"""

"""
#@tool
extends Node


var ws := WebSocketPeer.new()
var connected := false

func _ready():
    # Try connecting to the Python daemon
    var err = ws.connect_to_url("ws://localhost:8765")
    if err != OK:
        print("Connect error: ", err)
        
var last_header 

func _on_data():
    while ws.get_available_packet_count() > 0:
        var pkt = ws.get_packet()

        if ws.was_string_packet():
            var header = JSON.parse_string(pkt.get_string_from_utf8())
            print("Header:", header)
            last_header = header  # store it for the next binary frame

        else:
            var bytes = pkt
            print("Binary payload:", bytes.size(), "bytes")

            if last_header.type == "image":
                #_handle_image(bytes, last_header)
                pass


func _process(delta):
    ws.poll()

    match ws.get_ready_state():
        WebSocketPeer.STATE_CONNECTING:
            # Still trying
            pass

        WebSocketPeer.STATE_OPEN:
            if not connected:
                connected = true
                print("Connected to Python daemon")

                # Send a test message
                #var msg = {"cmd": "ping"}
                var msg = {"cmd": "ping_binary"}
                ws.send_text(JSON.stringify(msg))

            # Handle incoming packets
            while ws.get_available_packet_count() > 0:
                var pkt = ws.get_packet()

                if ws.was_string_packet():
                    print("Received text: ", pkt.get_string_from_utf8())
                else:
                    print("Received binary: ", pkt.size(), "bytes")

        WebSocketPeer.STATE_CLOSING:
            # Waiting for close handshake
            pass

        WebSocketPeer.STATE_CLOSED:
            if connected:
                connected = false
                print("Disconnected from daemon")
            # Optional: auto-reconnect
            _attempt_reconnect()

func _attempt_reconnect():
    await get_tree().create_timer(1.0).timeout
    print("Reconnecting...")
    ws = WebSocketPeer.new()
    ws.connect_to_url("ws://localhost:8765")
