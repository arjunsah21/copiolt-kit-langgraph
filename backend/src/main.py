import os
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.agent import LocalGuideAgent
from src.logger import logger, log_request, log_error

# Load environment variables
load_dotenv()

# Initialize Langfuse (optional - only if keys are provided)
langfuse_client = None

try:
    from langfuse import Langfuse

    if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
        langfuse_client = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        logger.info("Langfuse tracing enabled")
    else:
        logger.info("Langfuse tracing disabled (no keys provided)")
except Exception:
    logger.warning("Langfuse not available - tracing disabled")
    # Create a no-op decorator
    def observe(name=None):
        def decorator(func):
            return func
        return decorator


# Global agent instance
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global agent
    
    # Startup
    logger.info("Starting Local Guide API server...")
    
    # Initialize LLM
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment!")
        raise ValueError("OPENAI_API_KEY is required")
    
    # Get model from env or use default
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    logger.info(f"Initializing LLM with model: {model_name}")
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        api_key=openai_key
    )
    
    # Initialize agent
    agent = LocalGuideAgent(llm)
    logger.info("LocalGuideAgent initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local Guide API server...")
    if langfuse_client:
        langfuse_client.flush()


# Initialize FastAPI app
app = FastAPI(
    title="Local Guide API",
    description="AI agent for weather and restaurant recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# CopilotKit Protocol Models
class Message(BaseModel):
    role: str
    content: str = ""


class CopilotKitRequest(BaseModel):
    """CopilotKit/AG-UI compatible request format"""
    messages: list[Message] = []
    # Optional fields that CopilotKit might send
    tools: list = []
    context: list = []
    forwardedProps: dict = {}
    method: str | None = None  # CopilotKit sends 'agent/connect' initially
    state: dict = {}  # CopilotKit state management


def create_sse_event(event_type: str, data: dict) -> str:
    """Create a Server-Sent Event in CopilotKit format"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def stream_agent_response(
    request_data: CopilotKitRequest,
    latitude: float,
    longitude: float,
    thread_id_override: str | None = None,
    run_id_override: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream agent response in CopilotKit AG-UI protocol format"""
    
    try:
        # Get the last user message
        user_messages = [msg for msg in request_data.messages if msg.role == "user"]
        if not user_messages:
            yield create_sse_event("error", {"message": "No user message found"})
            return
        
        last_message = user_messages[-1].content
        
        # Generate unique IDs
        message_id = f"msg_{datetime.utcnow().timestamp()}"
        run_id = run_id_override or f"run_{datetime.utcnow().timestamp()}"
        thread_id = thread_id_override or request_data.forwardedProps.get(
            "threadId",
            f"thread_{datetime.utcnow().timestamp()}",
        )
        
        # Send RUN_STARTED event
        yield create_sse_event("run_started", {
            "type": "RUN_STARTED",
            "threadId": thread_id,
            "runId": run_id
        })
        
        # Run the agent
        result = await agent.run(
            user_message=last_message,
            latitude=latitude,
            longitude=longitude
        )
        
        response_text = result["response"]
        
        # Send TEXT_MESSAGE_START event
        yield create_sse_event("text_message_start", {
            "type": "TEXT_MESSAGE_START",
            "messageId": message_id,
            "role": "assistant"
        })
        
        # Stream the response text in chunks (simulating streaming)
        chunk_size = 50
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield create_sse_event("text_message_content", {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": message_id,
                "delta": chunk
            })
            await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        # Send TEXT_MESSAGE_END event
        yield create_sse_event("text_message_end", {
            "type": "TEXT_MESSAGE_END",
            "messageId": message_id
        })
        
        # Send RUN_FINISHED event
        yield create_sse_event("run_finished", {
            "type": "RUN_FINISHED",
            "threadId": thread_id,
            "runId": run_id
        })
        
    except Exception as e:
        log_error(e, "stream_agent_response")
        yield create_sse_event("error", {
            "type": "RUN_ERROR",
            "message": str(e)
        })


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Local Guide API",
        "langfuse_enabled": langfuse_client is not None
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "langfuse_enabled": langfuse_client is not None,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/api/copilotkit")
async def copilotkit_endpoint(request: Request):
    """
    CopilotKit-compatible endpoint using AG-UI protocol with SSE streaming.
    This endpoint works directly with CopilotKit's <CopilotKit> component.
    """
    try:
        # Parse request body
        body = await request.json()

        # Handle CopilotKit "single endpoint" runtime info request
        if body.get("method") == "info":
            return JSONResponse(
                {
                    "version": "local-dev",
                    "agents": {
                        "default": {
                            "name": "Local Guide Assistant",
                            "className": "LocalGuideAgent",
                            "description": "Helps with weather and restaurant recommendations.",
                        }
                    },
                    "audioFileTranscriptionEnabled": False,
                }
            )

        # Handle CopilotKit "single endpoint" agent stop
        if body.get("method") == "agent/stop":
            return JSONResponse({"status": "stopped"})

        # Unwrap "single endpoint" agent requests
        envelope_method = body.get("method")
        envelope_body = body.get("body") if isinstance(body.get("body"), dict) else None
        if envelope_method in ("agent/connect", "agent/run") and envelope_body is not None:
            request_payload = envelope_body
        else:
            request_payload = body

        request_data = CopilotKitRequest(**request_payload)

        log_request("/api/copilotkit", body)
        
        # Handle agent/connect method (CopilotKit handshake)
        if envelope_method == "agent/connect" or request_data.method == "agent/connect":
            thread_id = None
            run_id = None
            if isinstance(request_payload, dict):
                thread_id = request_payload.get("threadId")
                run_id = request_payload.get("runId")

            if not thread_id:
                thread_id = f"thread_{datetime.utcnow().timestamp()}"
            if not run_id:
                run_id = f"run_{datetime.utcnow().timestamp()}"

            async def connect_stream():
                yield create_sse_event(
                    "run_started",
                    {
                        "type": "RUN_STARTED",
                        "threadId": thread_id,
                        "runId": run_id,
                    },
                )
                yield create_sse_event(
                    "run_finished",
                    {
                        "type": "RUN_FINISHED",
                        "threadId": thread_id,
                        "runId": run_id,
                    },
                )
            
            return StreamingResponse(
                connect_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        if not agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        # Extract location from forwardedProps or use default
        forwarded_props = request_data.forwardedProps or {}
        location = forwarded_props.get("location", {})
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        
        # If no messages and we're not in single-endpoint mode, return ready event
        if not request_data.messages and envelope_method is None:
            async def ready_stream():
                yield create_sse_event("ready", {
                    "type": "READY",
                    "status": "awaiting_input"
                })

            return StreamingResponse(
                ready_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        if not latitude or not longitude:
            # Return error event
            error_msg = "Location coordinates required. Please enable location access."
            async def error_stream():
                yield create_sse_event("error", {
                    "type": "RUN_ERROR",
                    "message": error_msg
                })
            
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Stream response
        thread_id_override = request_payload.get("threadId") if isinstance(request_payload, dict) else None
        run_id_override = request_payload.get("runId") if isinstance(request_payload, dict) else None
        return StreamingResponse(
            stream_agent_response(
                request_data,
                latitude,
                longitude,
                thread_id_override=thread_id_override,
                run_id_override=run_id_override,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as err:
        log_error(err, "copilotkit_endpoint")
        
        # Return error stream - capture error in variable first
        error_message = str(err)
        async def error_stream():
            yield create_sse_event("error", {
                "type": "RUN_ERROR",
                "message": f"Server error: {error_message}"
            })
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
