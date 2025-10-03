from mcp import Tool, ToolResponse, Server

# from anthropic import Anthropic

# Si 'mcp' no existe, comenta las importaciones y usa un print simple para simular.
class SaludoTool(Tool):
    name = "saludo"
    description = "Devuelve un saludo personalizado"
    inputSchema = {"type": "object", "properties": {"nombre": {"type": "string"}}, "required": ["nombre"]}

    async def call(self, input):
        nombre = input.get("nombre", "amigo")
        return ToolResponse(content=f"Hola, {nombre}! ¿Cómo puedo ayudarte hoy?")


async def main():
    server = Server(tools=[SaludoTool()])
    await server.serve()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
