"""

Docstring for lib.python_daemon_test

pip install websockets # very small and standard library

"""


import asyncio
import json
import websockets


class ServerBase:

    port = 8765


    async def cmd_ping(self, websocket, data):
        await websocket.send(json.dumps({"reply": "pong"}))

    def __init__(self):
        # Track connected clients (optional but useful)
        self.clients = set()

        # Command dispatcher
        self.commands = {
            "ping": self.cmd_ping,
        }
    # -----------------------------
    # Connection Handler
    # -----------------------------

    async def handle_connection(self, websocket):
        print("Client connected:", websocket)
        self.clients.add(websocket)

        try:
            async for message in websocket:
                # Parse JSON
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "invalid_json"}))
                    continue

                # Dispatch command
                cmd = data.get("cmd")
                handler = self.commands.get(cmd)

                if handler:
                    await handler(websocket, data)
                else:
                    await websocket.send(json.dumps({"error": "unknown_command"}))

        except websockets.exceptions.ConnectionClosedError:
            print("Client disconnected cleanly")
        except ConnectionResetError:
            print("Client disconnected abruptly")
        finally:
            self.clients.remove(websocket)

    # -----------------------------
    # Main Server Loop
    # -----------------------------
    async def main(self):
        print(f"Python daemon listening on ws://localhost:{self.port}")
        async with websockets.serve(self.handle_connection, "localhost", self.port):
            await asyncio.Future()  # run forever


class Server(ServerBase):

    # -----------------------------
    # Command Handlers
    # -----------------------------
    async def cmd_ping(self, websocket, data):
        await websocket.send(json.dumps({"reply": "pong"}))

    async def cmd_ping_binary(self, websocket, data):
        # Example raw bytes (you can replace this with anything)
        raw = b"\x01\x02\x03\x04PINGDATA"

        await websocket.send(raw)



    async def cmd_send_image(self, websocket, data):
        # 1. Send a text header describing the binary payload
        header = {
            "type": "image",
            "format": "rgba",
            "width": 256,
            "height": 256,
            "label": "preview"
        }
        await websocket.send(json.dumps(header))

        # 2. Send the raw bytes
        raw = bytes([255, 0, 0, 255] * (256 * 256))  # example RGBA buffer
        await websocket.send(raw)



    async def cmd_send_array(self, websocket, data):
        import numpy as np

        arr = np.arange(256, dtype=np.uint8)
        await websocket.send(arr.tobytes())



    def __init__(self):
        super().__init__()

        # Command dispatcher
        self.commands = {
            "ping": self.cmd_ping,
            "ping_binary": self.cmd_ping_binary,
        }


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    asyncio.run(Server().main())
