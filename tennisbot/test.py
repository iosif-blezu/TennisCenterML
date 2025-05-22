import sys

from tennisbot.agent.router import get_router_agent

def main():
    agent = get_router_agent()
    if len(sys.argv) > 1:
        user_msg = " ".join(sys.argv[1:])
        print(agent.invoke({"input": user_msg})["output"])
    else:
        print("Type a message (empty line to quit):")
        while True:
            msg = input("HUMAN> ")
            if not msg.strip():
                break
            resp = agent.invoke({"input": msg})
            print("LLM", resp["output"])

if __name__ == "__main__":
    main()
