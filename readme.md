You now have a genuinely capable multi‑step agent, so the most useful sample questions are the ones that exercise routing, tool use, multi‑step reasoning, and model comparison. Grouping them helps you see the different behaviors your agent can perform.

## 🧭 Model‑routing questions

These trigger the agent to decide which model to call or to follow your explicit instruction.

“ask gpt‑4.1 what is 2+2”

“use gemini‑2.5-flash to summarize this paragraph”

“ask gpt‑5-mini to explain how transformers work”

“use gemini‑1.5-pro to translate this into French”

“ask openai for a short poem about Hoboken”

These test whether the agent correctly parses your intent and calls the right tool.

## 🔧 Multi‑step tool‑use questions

These require the agent to call a tool, get the result, then reason again.

“compare gpt‑4.1 and gemini‑2.5-flash for code generation”

“ask gpt‑4.1 to summarize this text, then ask gemini‑2.5-flash to critique the summary”

“use gemini‑2.0-pro to extract key points, then ask gpt‑4.1 to rewrite them more clearly”

“which model is better for long‑context summarization? show evidence”

These force the agent into a loop: think → tool → think → answer.

## 📚 Model‑listing and capability questions

These test your list_openai_models and list_gemini_models tools.

“show me all openai models you support”

“list gemini models”

“which models support vision?”

“which models are best for reasoning tasks?”

These confirm that your model registry is wired correctly.

## 🧠 Reasoning + tool use

These test whether the agent chooses to call a tool even when not explicitly told.

“who is faster at math: gpt‑4.1 or gemini‑2.5-flash?”

“which model writes better python code?”

“summarize the differences between openai and gemini models”

“which model should I use for legal document summarization?”

These require the agent to gather evidence via tools before answering.

## 🧩 Mixed‑mode questions (your agent’s strongest area)

These combine reasoning, routing, and tool use.

“ask gpt‑4.1 to generate a short story, then ask gemini‑2.5-flash to rewrite it funnier”

“use gemini‑2.0-pro to extract entities from this text, then ask gpt‑4.1 to classify them”

“compare the answers from gpt‑4.1 and gemini‑2.5-flash to the question: ‘what is the future of AI?’”

“which model gives more detailed answers? test it with a question about quantum computing”

These show off the multi‑step LangGraph loop you built.

## 🛠️ Debugging / transparency questions

These help you verify tool‑call tracking.

“what tools did you use to answer my last question?”

“show me the tool calls you made”

“explain why you chose that model”

These confirm your tool_calls tracking is working.

If you want, I can help you craft a small demo script of 10–15 questions that systematically exercise every part of your agent (routing, multi‑step loops, model comparison, memory, etc.).