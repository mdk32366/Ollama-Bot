import os
import json
from typing import List, TypedDict

from langchain_openai import ChatOpenAI
from openai import OpenAI

from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

from langgraph.graph import StateGraph, END


# -----------------------------
# Model registries
# -----------------------------

OPENAI_MODELS = {
    "chat": [
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano"
    ],
    "image": [
        "dall-e-2",
        "dall-e-3"
    ],
}

GEMINI_MODELS = {
    "chat": [
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview-09-2025"
    ],

}


# -----------------------------
# Tool implementations
# -----------------------------

def ask_openai(model: str, question: str) -> str:
    print(f"ask_openai called with model={model} and question={question}")
    llm = ChatOpenAI(model=model)
    resp = llm.invoke(question)
    print(f"ask_openai response: {resp.content}!")
    return resp.content


def ask_gemini(model: str, question: str) -> str:
    print(f"ask_gemini called with model={model} and question={question}")
    llm = ChatGoogleGenerativeAI(model=model)
    resp = llm.invoke(question)
    print(f"ask_gemini response: {resp.content}!")
    return resp.content


def list_openai_models() -> str:
    lines = ["Available OpenAI models:"]
    for category, models in OPENAI_MODELS.items():
        lines.append(f"\n{category}:")
        for m in models:
            lines.append(f"  - {m}")
    return "\n".join(lines)


def list_gemini_models() -> str:
    lines = ["Available Gemini models:"]
    for family, models in GEMINI_MODELS.items():
        lines.append(f"\n{family}:")
        for m in models:
            lines.append(f"  - {m}")
    return "\n".join(lines)

def generate_openai_image(model: str, prompt: str) -> str:
    try:
        client = OpenAI()
        result = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024"
        )

        if not result.data or not result.data[0].b64_json:
            return "__IMAGE_FAILED__"

        return "data:image/png;base64," + result.data[0].b64_json

    except Exception as e:
        return "__IMAGE_FAILED__"


# -----------------------------
# LangGraph state definition
# -----------------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]
    done: bool
    tool_calls: List[str]


# -----------------------------
# Engine class
# -----------------------------

class OllamaChatEngine:
    """
    Hybrid multi-step agent using LangGraph.

    - Ollama is the reasoning model.
    - Tools:
        - ask_openai(model, question)
        - ask_gemini(model, question)
        - list_openai_models()
        - list_gemini_models()
    - The agent can:
        - answer directly with Ollama
        - call tools via a TOOL_CALL protocol
        - loop: think → call tool → think → answer
    """

    def __init__(self, model_name: str = "qwen2.5-coder:latest"):
        # Load config.json
        config_path = os.path.join(os.path.dirname(__file__), "configs.json")
        print(config_path)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # Set environment variables for OpenAI + Gemini
            if "OPENAI_API_KEY" in cfg:
                os.environ["OPENAI_API_KEY"] = cfg["OPENAI_API_KEY"]

            if "GOOGLE_API_KEY" in cfg:
                os.environ["GOOGLE_API_KEY"] = cfg["GOOGLE_API_KEY"]

        # Continue with your existing initialization
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)
        self.history: List[dict] = []

        # Build LangGraph once
        self.app = self._build_graph()

    # -------------------------
    # Graph construction
    # -------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("agent", self._agent_node)
        graph.add_node("router", self._router_node)

        graph.set_entry_point("agent")
        graph.add_edge("agent", "router")
        graph.add_conditional_edges(
            "router",
            self._should_continue,
            {
                "agent": "agent",
                "end": END,
            },
        )

        return graph.compile()

    # -------------------------
    # Graph nodes
    # -------------------------

    def _agent_node(self, state: AgentState) -> AgentState:
        """
        Agent node: uses Ollama to either:
        - answer directly, or
        - emit a TOOL_CALL instruction.

        TOOL_CALL format (must be in the last AI message content):

        TOOL_CALL: <tool_name>
        ARGS: {"model": "...", "question": "..."}

        Valid tool_name values:
        - ask_openai
        - ask_gemini
        - list_openai_models
        - list_gemini_models
        """

        system = SystemMessage(
            content=(
                "You are a routing and reasoning agent.\n"
                "You can answer directly using your own reasoning, or you can call tools.\n\n"

                "TOOLS:\n"
                "1) ask_openai: Ask an OpenAI model a question.\n"
                "   ARGS: {\"model\": \"<model-name>\", \"question\": \"<question>\"}\n\n"
                "2) ask_gemini: Ask a Gemini model a question.\n"
                "   ARGS: {\"model\": \"<model-name>\", \"question\": \"<question>\"}\n\n"
                "3) list_openai_models: List available OpenAI models.\n"
                "   ARGS: {}\n\n"
                "4) list_gemini_models: List available Gemini models.\n"
                "   ARGS: {}\n\n"
                "5) generate_openai_image: Generate an image using an OpenAI image model.\n"
                "  ARGS: {\"model\": \"<model-name>\", \"prompt\": \"<text>\"}\n\n"
                "6) generate_gemini_image: Generate an image using a Gemini image model.\n"
                "  ARGS: {\"model\": \"<model-name>\", \"prompt\": \"<text>\"}\n\n"

                "When you call an image tool, return ONLY:\n"
                "TOOL_CALL: <tool_name>\n"
                "ARGS: {\"model\": \"...\", \"prompt\": \"...\"}\n\n"

                "If a tool returns \"__IMAGE_FAILED__\", do NOT describe the image.\n"
                "Instead, say: \"Image generation failed. Please try a different model or prompt.\"\n"

                "When you decide to call a tool, respond ONLY in this format:\n"
                "TOOL_CALL: <tool_name>\n"
                "ARGS: { ... }\n\n"

                "IMPORTANT:\n"
                "When you receive a message that begins with 'Result from <tool_name>:', "
                "you MUST use that result to produce the final answer to the user. "
                "Do NOT stop after a tool call. Always continue reasoning until you "
                "produce a final natural-language answer with no TOOL_CALL.\n"
            )
        )

        messages = [system] + state["messages"]
        response = self.llm.invoke(messages)

        new_messages = state["messages"] + [response]
        done = "TOOL_CALL:" not in response.content

        return {"messages": new_messages, "done": done}

    def _router_node(self, state: AgentState) -> AgentState:
        """
        Router node:
        - If last AI message contains TOOL_CALL, parse and execute the tool.
        - Append tool result as an AI message.
        - Mark done=False so agent can think again with tool result.
        - If no TOOL_CALL, mark done=True and end.
        """

        if not state["messages"]:
            return {**state, "done": True}

        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {**state, "done": True}

        content = last.content
        if "TOOL_CALL:" not in content:
            # No tool call → we are done
            return {**state, "done": True}

        # Parse TOOL_CALL
        lines = content.splitlines()
        tool_name = None
        args_json = "{}"

        for line in lines:
            line = line.strip()
            if line.startswith("TOOL_CALL:"):
                tool_name = line.split("TOOL_CALL:")[1].strip()
            elif line.startswith("ARGS:"):
                args_json = line.split("ARGS:")[1].strip()

        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            tool_result = "Tool call failed: invalid ARGS JSON."
            new_msg = AIMessage(content=tool_result)
            return {
                "messages": state["messages"] + [new_msg],
                "done": True,
            }

        tool_result = self._execute_tool(tool_name, args)

        call_record = f"{tool_name}({json.dumps(args)})"
        new_tool_calls = state["tool_calls"] + [call_record]

        tool_msg = HumanMessage(content=f"Tool result: {tool_result}")
        new_messages = state["messages"] + [tool_msg]

        return {
            "messages": new_messages,
            "done": False,
            "tool_calls": new_tool_calls
        }


    def _should_continue(self, state: AgentState) -> str:
        return "end" if state["done"] else "agent"

    # -------------------------
    # Tool dispatcher
    # -------------------------

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        try:
            if tool_name == "ask_openai":
                model = args.get("model")
                question = args.get("question")
                if not model or not question:
                    return "ask_openai requires 'model' and 'question'."
                return ask_openai(model, question)

            if tool_name == "ask_gemini":
                model = args.get("model")
                question = args.get("question")
                if not model or not question:
                    return "ask_gemini requires 'model' and 'question'."
                return ask_gemini(model, question)

            if tool_name == "list_openai_models":
                return list_openai_models()

            if tool_name == "list_gemini_models":
                return list_gemini_models()
            if tool_name == "generate_openai_image":
                model = args.get("model")
                prompt = args.get("prompt")
                if not model or not prompt:
                    return "generate_openai_image requires 'model' and 'prompt'."
                return generate_openai_image(model, prompt)

            if tool_name == "generate_gemini_image":
                model = args.get("model")
                prompt = args.get("prompt")
                if not model or not prompt:
                    return "generate_gemini_image requires 'model' and 'prompt'."
                return generate_gemini_image(model, prompt)

            return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool execution error for {tool_name}: {e}"

    # -------------------------
    # Public API
    # -------------------------
    def ask(self, prompt: str) -> dict:
        messages = []

        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=prompt))

        final_state = self.app.invoke({
            "messages": messages,
            "done": False,
            "tool_calls": []
        })

        final_messages = final_state["messages"]
        tool_calls = final_state["tool_calls"]

        answer_text = ""
        image_base64 = None

        for m in reversed(final_messages):
            if isinstance(m, AIMessage) and "TOOL_CALL:" not in m.content:
                answer_text = m.content
                break

        # Detect image
        if answer_text.startswith("data:image"):
            image_base64 = answer_text
            answer_text = "(Image generated below)"
        #GA: added in tool_calls here to fix bug.
        self.history.append({"role": "assistant", "content": prompt})
        assistant_text = answer_text

        if tool_calls:
            assistant_text += "\n\n### Tools used:\n" + "\n".join(f"- `{t}`" for t in tool_calls)

        self.history.append({"role": "assistant", "content": assistant_text,"tools": tool_calls})

        return {
            "text": answer_text,
            "image": image_base64,
            "tools": tool_calls
        }
