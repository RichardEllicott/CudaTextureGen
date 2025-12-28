"""

websocket server

requires:
pip install websockets

"""


import asyncio
import json
import websockets
import inspect
from websockets.asyncio.server import ServerConnection


class ServerBase:
    """
    server base

    commands are detected by reflection, must follow form:

    async def CMD_PING(self, websocket: ServerConnection, data) -> None:

    """
    port: int = 8765  # default port
    clients: set[ServerConnection] = set()  # connected clients

    async def CMD_PING(self, websocket: ServerConnection, data) -> None:
        """
        command example
        """
        await websocket.send(json.dumps({"reply": "pong"}))

    async def handle_connection(self, websocket: ServerConnection) -> None:
        """
        connection handler
        """
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

                cmd: str = data.get("cmd")  # command string
                cmd = cmd.capitalize()  # ensure capitals
                handler = getattr(self, f"CMD_{cmd}", None)  # get function

                if inspect.iscoroutinefunction(handler):  # check is an aysync def
                    await handler(websocket, data)
                else:
                    await websocket.send(json.dumps({"error": "unknown_command"}))

        except websockets.exceptions.ConnectionClosedError:
            print("Client disconnected cleanly")
        except ConnectionResetError:
            print("Client disconnected abruptly")
        finally:
            self.clients.remove(websocket)

    async def main(self) -> None:
        """
        main server loop
        """
        print(f"Python daemon listening on ws://localhost:{self.port}")
        async with websockets.serve(self.handle_connection, "localhost", self.port):
            await asyncio.Future()  # run forever

    def launch(self):
        """
        launch the server (will block thread)
        """
        asyncio.run(self.main())


class Server(ServerBase):

    async def CMD_PING(self, websocket: ServerConnection, data) -> None:

        print("cmd_ping data:", data)

        await websocket.send(json.dumps({"reply": "pong"}))

    async def CMD_PING_BINARY(self, websocket, data) -> None:

        print("cmd_ping_binary data:", data)

        # Example raw bytes (you can replace this with anything)
        raw = b"\x01\x02\x03\x04PINGDATA"

        await websocket.send(raw)

    async def CMD_SEND_ARRAY_TEST(self, websocket: ServerConnection, data) -> None:
        """
        example:
        sending back a 32 bit numpy array (gradient) with a header, then the bytes 
        """
        import numpy as np

        width: int = 32
        height: int = 32
        if isinstance(data, dict):
            width = int(data.get("width", width))
            height = int(data.get("height", height))

        header = {
            "type": "image",
            "format": "f32",
            "width": width,
            "height": height,
            "label": "preview"
        }
        await websocket.send(json.dumps(header))  # send header

        arr = np.linspace(0.0, 1.0, height*width, dtype=np.float32).reshape((height, width))  # Create a 2D float32 array gradient in range [0, 1]
        await websocket.send(arr.tobytes())  # send array bytes






# launch example
if __name__ == "__main__":
    Server().launch()
