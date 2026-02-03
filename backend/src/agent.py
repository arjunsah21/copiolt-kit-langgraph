import json
import os
import time
import uuid
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import operator

from src.tools import get_weather, get_nearby_restaurants
from src.logger import log_agent_start, log_error
from langchain_core.callbacks import BaseCallbackHandler

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except Exception:
    LangfuseCallbackHandler = None

DEBUG_LOG_PATH = "/Users/arjun.kumar/learnings/genai/copilot-kit/.cursor/debug.log"


def _debug_log(payload: dict) -> None:
    try:
        payload["timestamp"] = int(time.time() * 1000)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload) + "\n")
    except Exception:
        pass


class DebugCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, run_id, parent_run_id=None, **kwargs):
        input_keys = list(inputs.keys()) if isinstance(inputs, dict) else type(inputs).__name__
        # region agent log
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "B",
                "location": "agent.py:DebugCallbackHandler.on_chain_start",
                "message": "chain_start",
                "data": {
                    "runId": str(run_id),
                    "parentRunId": str(parent_run_id) if parent_run_id else None,
                    "name": (serialized or {}).get("name"),
                    "inputKeys": input_keys,
                },
            }
        )
        # endregion


def _summarize_outputs(outputs) -> dict | None:
    if isinstance(outputs, dict):
        messages = outputs.get("messages")
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content:
                return {"message": content}
    return None


if LangfuseCallbackHandler:
    class LocalLangfuseCallbackHandler(LangfuseCallbackHandler):
        def __init__(self, *, trace_context, update_trace, trace_name, input_summary):
            super().__init__(trace_context=trace_context, update_trace=update_trace)
            self._trace_name = trace_name
            self._input_summary = input_summary

        def get_langchain_run_name(self, serialized, **kwargs):
            name = super().get_langchain_run_name(serialized, **kwargs)
            if name == "<unknown>":
                run_name = kwargs.get("run_name")
                if run_name:
                    return str(run_name)
            return name

        def on_chain_start(self, serialized, inputs, run_id, parent_run_id=None, **kwargs):
            super().on_chain_start(
                serialized,
                inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
                **kwargs,
            )
            if parent_run_id is None:
                span = self.runs.get(run_id)
                if span:
                    span.update_trace(
                        name=self._trace_name,
                        input=self._input_summary,
                    )
                    # region agent log
                    _debug_log(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "D",
                            "location": "agent.py:LocalLangfuseCallbackHandler.on_chain_start",
                            "message": "trace_update_input",
                            "data": {
                                "traceName": self._trace_name,
                                "hasInputSummary": True,
                            },
                        }
                    )
                    # endregion

        def on_chain_end(self, outputs, run_id, parent_run_id=None, **kwargs):
            super().on_chain_end(
                outputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
                **kwargs,
            )
            if parent_run_id is None:
                summary = _summarize_outputs(outputs)
                if summary:
                    span = self.runs.get(run_id)
                    if span:
                        span.update_trace(output=summary)
                        # region agent log
                        _debug_log(
                            {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "D",
                                "location": "agent.py:LocalLangfuseCallbackHandler.on_chain_end",
                                "message": "trace_update_output",
                                "data": {"hasOutputSummary": True},
                            }
                        )
                        # endregion

    def on_chain_end(self, outputs, run_id, parent_run_id=None, **kwargs):
        output_keys = list(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__
        # region agent log
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "C",
                "location": "agent.py:DebugCallbackHandler.on_chain_end",
                "message": "chain_end",
                "data": {
                    "runId": str(run_id),
                    "parentRunId": str(parent_run_id) if parent_run_id else None,
                    "outputKeys": output_keys,
                },
            }
        )
        # endregion


def _check_langfuse_connection(host: str, timeout: float = 2.0) -> bool:
    """
    Check if Langfuse is reachable at the given host.
    
    Args:
        host: Langfuse host URL (e.g., "http://localhost:3000")
        timeout: Connection timeout in seconds
    
    Returns:
        True if Langfuse is reachable, False otherwise
    """
    import httpx
    from src.logger import logger
    
    try:
        # Try to reach the Langfuse health endpoint or root
        response = httpx.get(f"{host}/api/public/health", timeout=timeout)
        return response.status_code < 500
    except Exception:
        # If connection fails, Langfuse is not reachable
        logger.info(f"Langfuse is not running at {host} - skipping tracing")
        return False


def _build_langfuse_callbacks(
    user_message: str,
    latitude: float,
    longitude: float,
) -> tuple[list, str | None]:
    callbacks: list = [DebugCallbackHandler()]
    if LangfuseCallbackHandler is None:
        # region agent log
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "agent.py:_build_langfuse_callbacks",
                "message": "langfuse_handler_unavailable",
                "data": {"reason": "import_failed"},
            }
        )
        # endregion
        return callbacks, None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        # region agent log
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "agent.py:_build_langfuse_callbacks",
                "message": "langfuse_handler_unavailable",
                "data": {"reason": "missing_keys"},
            }
        )
        # endregion
        return callbacks, None

    # Check if Langfuse is reachable
    langfuse_host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if not _check_langfuse_connection(langfuse_host):
        # region agent log
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "agent.py:_build_langfuse_callbacks",
                "message": "langfuse_handler_unavailable",
                "data": {"reason": "connection_failed", "host": langfuse_host},
            }
        )
        # endregion
        return callbacks, None

    trace_id = uuid.uuid4().hex
    # Create a descriptive trace name with user message preview
    message_preview = user_message[:50] + "..." if len(user_message) > 50 else user_message
    trace_name = f"Local Guide: {message_preview}"
    input_summary = {
        "message": user_message,
        "latitude": latitude,
        "longitude": longitude,
    }
    
    try:
        handler = LocalLangfuseCallbackHandler(
            trace_context={"trace_id": trace_id},
            update_trace=True,
            trace_name=trace_name,
            input_summary=input_summary,
        )
        callbacks.append(handler)
        # region agent log
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "agent.py:_build_langfuse_callbacks",
                "message": "langfuse_handler_ready",
                "data": {
                    "callbacksCount": len(callbacks),
                    "traceIdSet": bool(trace_id),
                    "updateTrace": True,
                },
            }
        )
        # endregion
        return callbacks, trace_id
    except Exception as e:
        # If handler creation fails, log and continue without tracing
        from src.logger import logger
        logger.info(f"Failed to create Langfuse handler: {e} - skipping tracing")
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "agent.py:_build_langfuse_callbacks",
                "message": "langfuse_handler_creation_failed",
                "data": {"error": str(e)},
            }
        )
        return callbacks, None



class AgentState(TypedDict):
    """State of the agent workflow"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    latitude: float | None
    longitude: float | None
    intent: str | None
    result: dict | None


# Define tools using LangChain's @tool decorator
@tool
async def fetch_weather(latitude: float, longitude: float) -> dict:
    """
    Fetch current weather conditions for a given location.
    
    Args:
        latitude: The latitude coordinate of the location
        longitude: The longitude coordinate of the location
    
    Returns:
        Weather data including temperature, conditions, humidity, etc.
    """
    return await get_weather(latitude, longitude)


@tool
async def fetch_restaurants(latitude: float, longitude: float, radius_meters: int = 1000) -> list:
    """
    Find restaurants near a given location.
    
    Args:
        latitude: The latitude coordinate of the location
        longitude: The longitude coordinate of the location
        radius_meters: Search radius in meters (default 1000)
    
    Returns:
        List of nearby restaurants with details
    """
    return await get_nearby_restaurants(latitude, longitude, radius_meters)


class LocalGuideAgent:
    """LangGraph agent for local guide (weather + restaurants)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [fetch_weather, fetch_restaurants]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with conditional routing"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("fetch_weather", self._fetch_weather_node)
        workflow.add_node("fetch_restaurants", self._fetch_restaurants_node)
        workflow.add_node("format_response", self._format_response)
        
        # Define edges
        workflow.set_entry_point("classify_intent")
        
        # Conditional edge based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "weather": "fetch_weather",
                "restaurants": "fetch_restaurants",
            }
        )
        
        # Both tool nodes go to format_response
        workflow.add_edge("fetch_weather", "format_response")
        workflow.add_edge("fetch_restaurants", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def _route_by_intent(self, state: AgentState) -> str:
        """Route to the appropriate tool node based on classified intent"""
        intent = state.get("intent", "weather")
        return intent if intent in ["weather", "restaurants"] else "weather"
    
    async def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent (weather or restaurants)"""
        try:
            last_message = state["messages"][-1].content if state["messages"] else ""
            classifier = "keyword"
            
            # Simple keyword-based classification
            lower_msg = last_message.lower()
            
            if any(keyword in lower_msg for keyword in ["weather", "temperature", "rain", "forecast", "climate"]):
                intent = "weather"
            elif any(keyword in lower_msg for keyword in ["restaurant", "food", "eat", "dining", "cuisine"]):
                intent = "restaurants"
            else:
                # Default to asking LLM to classify
                classification_prompt = f"Classify this request as either 'weather' or 'restaurants': {last_message}"
                response = await self.llm.ainvoke([HumanMessage(content=classification_prompt)])
                intent = "weather" if "weather" in response.content.lower() else "restaurants"
                classifier = "llm"

            log_agent_start(
                last_message,
                state.get("latitude"),
                state.get("longitude"),
            )
            
            return {**state, "intent": intent}
        except Exception as e:
            log_error(e, "_classify_intent")
            return {**state, "intent": "weather"}  # Default fallback
    
    async def _fetch_weather_node(self, state: AgentState) -> AgentState:
        """Fetch weather data from Open-Meteo API"""
        try:
            latitude = state.get("latitude")
            longitude = state.get("longitude")
            
            if not latitude or not longitude:
                return {**state, "result": {"error": "Location coordinates required"}}
            
            result = await fetch_weather.ainvoke({"latitude": latitude, "longitude": longitude})
            return {**state, "result": result}
        except Exception as e:
            log_error(e, "_fetch_weather_node")
            return {**state, "result": {"error": str(e)}}
    
    async def _fetch_restaurants_node(self, state: AgentState) -> AgentState:
        """Fetch nearby restaurants from Overpass API"""
        try:
            latitude = state.get("latitude")
            longitude = state.get("longitude")
            
            if not latitude or not longitude:
                return {**state, "result": {"error": "Location coordinates required"}}
            
            result = await fetch_restaurants.ainvoke({"latitude": latitude, "longitude": longitude})
            return {**state, "result": result}
        except Exception as e:
            log_error(e, "_fetch_restaurants_node")
            return {**state, "result": {"error": str(e)}}
    
    async def _format_response(self, state: AgentState) -> AgentState:
        """Format the final response as an AI message"""
        try:
            result = state.get("result", {})
            intent = state.get("intent")
            branch = "unknown"
            
            if "error" in result:
                branch = "error"
                response_text = f"Sorry, I encountered an error: {result['error']}"
            elif intent == "weather":
                branch = "weather"
                temp = result.get("temperature")
                feels_like = result.get("feels_like")
                conditions = result.get("conditions")
                humidity = result.get("humidity")
                wind = result.get("wind_speed")
                
                response_text = f"""🌡️ **Current Weather**

**Temperature:** {temp}°C (feels like {feels_like}°C)
**Conditions:** {conditions}
**Humidity:** {humidity}%
**Wind Speed:** {wind} km/h

The weather is currently {conditions.lower()} with a temperature of {temp}°C."""
            else:  # restaurants
                if isinstance(result, list) and len(result) > 0:
                    branch = "restaurants_list"
                    response_text = "🍽️ **Nearby Restaurants**\n\n"
                    for i, restaurant in enumerate(result[:10], 1):
                        if "error" not in restaurant:
                            name = restaurant.get("name", "Unknown")
                            cuisine = restaurant.get("cuisine", "Not specified")
                            address = restaurant.get("address", "")
                            response_text += f"{i}. **{name}**\n"
                            response_text += f"   - Cuisine: {cuisine}\n"
                            if address:
                                response_text += f"   - Address: {address}\n"
                            response_text += "\n"
                else:
                    branch = "restaurants_empty"
                    response_text = "No restaurants found nearby."

            messages = [AIMessage(content=response_text)]
            
            return {**state, "messages": messages}
        except Exception as e:
            log_error(e, "_format_response")
            error_msg = AIMessage(content=f"Error formatting response: {str(e)}")
            return {**state, "messages": [error_msg]}
    
    async def run(self, user_message: str, latitude: float, longitude: float) -> dict:
        """
        Run the agent with a user message and location.
        
        Args:
            user_message: User's question/request
            latitude: User's latitude
            longitude: User's longitude
        
        Returns:
            Agent response
        """
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "latitude": latitude,
            "longitude": longitude,
            "intent": None,
            "result": None
        }
        
        callbacks, trace_id = _build_langfuse_callbacks(
            user_message,
            latitude,
            longitude,
        )
        # region agent log
        _debug_log(
            {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "agent.py:run",
                "message": "graph_invoke_start",
                "data": {
                    "callbacksCount": len(callbacks),
                    "traceIdSet": bool(trace_id),
                    "messageLength": len(user_message),
                },
            }
        )
        # endregion
        if callbacks and trace_id:
            final_state = await self.graph.ainvoke(
                initial_state,
                config={
                    "callbacks": callbacks,
                    "run_name": trace_id,
                },
            )
        else:
            final_state = await self.graph.ainvoke(initial_state)
        
        # Extract the response message
        response_messages = final_state.get("messages", [])
        response_text = response_messages[-1].content if response_messages else "No response generated"
        
        return {
            "response": response_text,
            "intent": final_state.get("intent"),
            "result": final_state.get("result")
        }
