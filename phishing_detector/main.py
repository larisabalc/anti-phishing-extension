import asyncio
import websockets
from webservice.websocket_handler import WebSocketHandler
#ssl_context = ssl.create_default_context()
#ssl_context.load_verify_locations(certifi.where())

async def main():
    server = await websockets.serve(WebSocketHandler.handle_connection, "localhost" , 9999) 
    await server.wait_closed()

asyncio.run(main())