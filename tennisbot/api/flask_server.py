# Create a new file, e.g., tennisbot/api/server_flask.py
# pip install Flask
from flask import Flask, request, jsonify
from tennisbot.agent.router import get_router_agent
from tennisbot.config import get_settings
import os

app = Flask(__name__)

# Load config and agent once
print("FLASK_SERVER: Loading settings...")
cfg = get_settings()  # This should set os.environ keys
print(f"FLASK_SERVER: cfg.OPENAI_API_KEY: {'SET' if cfg.OPENAI_API_KEY else 'NOT SET'}")
print(f"FLASK_SERVER: os.environ['OPENAI_API_KEY']: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")

print("FLASK_SERVER: Loading agent...")
agent = get_router_agent()
print("FLASK_SERVER: Agent loaded.")


@app.route("/chat", methods=["POST"])
def chat_handler():
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Input not provided"}), 400

    user_input = data["input"]
    try:
        response = agent.invoke({"input": user_input})
        return jsonify({"output": response.get("output", str(response))})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)  # Use a different port like 8002