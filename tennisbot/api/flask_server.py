import os
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

from flask import Flask, request, jsonify
from tennisbot.agent.router import get_router_agent
from tennisbot.config import get_settings

app = Flask(__name__)

print("FLASK_SERVER: Loading settings…")
cfg = get_settings()
print(f"FLASK_SERVER: OPENAI_API_KEY → {'SET' if cfg.OPENAI_API_KEY else 'NOT SET'}")
print(f"FLASK_SERVER: TAVILY_API_KEY → {'SET' if cfg.TAVILY_API_KEY else 'NOT SET'}")

print("FLASK_SERVER: Instantiating router agent…")
agent = get_router_agent()
print("FLASK_SERVER: Agent ready.")

@app.route("/chat", methods=["POST"])
def chat_handler():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("input", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # invoke returns a dict with "output" key
        result = agent.invoke({"input": user_input})

        # print memory
        mem = agent.memory.load_memory_variables({})
        print("===== CHAT MEMORY =====")
        for msg in mem["chat_history"]:
            # each msg is a HumanMessage or AIMessage
            print(f"{type(msg).__name__}: {msg.content}")
        print("=======================")

        out = result.get("output") if isinstance(result, dict) else str(result)
        return jsonify({"output": out})
    except Exception as e:

        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)