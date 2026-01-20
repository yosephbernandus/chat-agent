import asyncio
import websockets
import json


async def test_websocket():
    uri = "ws://localhost:8000/ws/chat"

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")

        # Test 1: RAG mode
        print("Test 1: RAG mode")
        await websocket.send(json.dumps({"message": "What is RAG?", "mode": "rag"}))

        response = ""
        while True:
            data = json.loads(await websocket.recv())

            if data["type"] == "sources":
                print(f"Sources: {len(data['data'])} documents")
            elif data["type"] == "token":
                response += data["data"]
                print(data["data"], end="", flush=True)
            elif data["type"] == "done":
                print("\n Response complete")
                break

        # Test 2: Follow-up question (tests history)
        print("\nTest 2: Follow-up question")
        await websocket.send(
            json.dumps(
                {"message": "Can you summarize that in one sentence?", "mode": "rag"}
            )
        )

        response2 = ""
        while True:
            data = json.loads(await websocket.recv())

            if data["type"] == "sources":
                print(f"Sources: {len(data['data'])} documents")
            elif data["type"] == "token":
                response2 += data["data"]
                print(data["data"], end="", flush=True)
            elif data["type"] == "done":
                print("\nResponse complete")
                break

        print("\nWebSocket test complete!")


if __name__ == "__main__":
    asyncio.run(test_websocket())
