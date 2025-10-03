from cliente import MCPClient

async def main():
    server_script_path = "server.py"  # Ruta relativa al servidor (ajusta si es necesario, ej: "../server.py")
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()
