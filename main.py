
from fastapi import FastAPI
from pydantic import BaseModel

# Import our existing multi-agent LangGraph workflow.
from app.agents.graph import graph

# Convert Python dictionaries into JSON SSE payloads.
import json

# FastAPI's streaming HTTP response.
from fastapi.responses import StreamingResponse

# Serve the browser interface as a static HTML file.
from fastapi.responses import FileResponse


# Create the FastAPI application.
app = FastAPI(
    title="Developer Documentation Assistant",
    version="1.0.0",
)


# Define the expected structure of incoming chat requests.
class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    # Return our vanilla HTML/JavaScript chat interface.
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    # Lightweight endpoint for checking server availability.
    # This does not test external LLM or MCP connectivity.
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    # Initialize a fresh LangGraph state for each request.
    initial_state = {
        "question": request.question,
        "selected_agent": "research",
        "answer": "",
        "trace": [],
        "agent_results": {},
    }

    # Execute the supervisor and specialized agents.
    result = graph.invoke(initial_state)

    # Return the answer and inspectable execution information.
    return {
        "answer": result["answer"],
        "plan": result.get("plan", []),
        "routing": result.get("routing"),
        "trace": result["trace"],
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    # Create a fresh state for the user's request.
    initial_state = {
        "question": request.question,
        "selected_agent": "research",
        "answer": "",
        "trace": [],
        "agent_results": {},
    }

    def event_stream():
        # Format a message according to the SSE protocol.
        def emit(event_type: str, data: dict) -> str:
            payload = json.dumps(data, ensure_ascii=False)
            return f"event: {event_type}\ndata: {payload}\n\n"

        try:
            # Tell the browser that processing has started.
            yield emit("status", {"message": "Processing question..."})

            # Execute LangGraph and receive node updates.
            # Each update arrives after a graph node finishes.
            final_state = dict(initial_state)

            for update in graph.stream(
                initial_state,
                stream_mode="updates",
            ):
                for node_name, changes in update.items():
                    # Merge each node's state updates.
                    final_state.update(changes)

                    # Report which node just completed.
                    yield emit("trace", {
                        "node": node_name,
                        "message": f"{node_name} completed",
                    })

            # Send the completed answer in small text chunks.
            # This tests SSE delivery, NOT actual LLM token streaming.
            answer = final_state.get("answer", "")

            for start in range(0, len(answer), 40):
                yield emit("token", {
                    "text": answer[start:start + 40],
                })

            # Send final routing and trace information.
            yield emit("done", {
                "routing": final_state.get("routing"),
                "trace": final_state.get("trace", []),
            })

        except Exception as error:
            # Report failures without exposing internal credentials.
            yield emit("error", {
                "message": f"Processing failed: {type(error).__name__}",
            })

    # Return an SSE response instead of ordinary JSON.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

