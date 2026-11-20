import ast
from datetime import datetime
import json
import operator
import urllib.request

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
LOG_FILE = "chat_history.log"

MODEL_NAME = "qwen2.5:1.5b"
SYSTEM_INSTRUCTION = (
    "You are Nebula AI, an intelligent, structured, and reliable assistant. "
    "Keep answers clear and accurate."
)

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node):
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_ast_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _eval_ast_node(node.left), _eval_ast_node(node.right)
        )
    raise ValueError("Unsupported mathematical expression.")


def calculate_expression(expression: str) -> str:
    try:
        return str(_eval_ast_node(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"Calculation error: {exc}"


def append_log(entry: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {entry}\n")


def list_installed_models() -> list[str]:
    req = urllib.request.Request(OLLAMA_TAGS_URL)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def stream_ollama(messages: list[dict], model_name: str) -> str:
    payload = json.dumps(
        {"model": model_name, "messages": messages, "stream": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    chunks = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            token = data.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                chunks.append(token)
    print("\n")
    return "".join(chunks)


def run_nebula() -> None:
    current_model = MODEL_NAME
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    append_log(f"--- Session started (model={current_model}) ---")
    print(
        f"Nebula AI ({current_model} via Ollama) is online. "
        "Commands: calc <expr>, /models, /model <name>, /clear, exit\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            append_log("--- Session ended ---")
            print("\nNebula AI: Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"exit", "quit"}:
            append_log("--- Session ended ---")
            print("Nebula AI: Goodbye!")
            break

        if cmd == "/clear":
            messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
            append_log("SYSTEM: Conversation memory cleared.")
            print("\nNebula AI: Conversation memory cleared.\n")
            continue

        if cmd == "/models":
            try:
                installed = list_installed_models()
                print("\nInstalled Ollama models:")
                for m in installed:
                    marker = " (active)" if m == current_model else ""
                    print(f"  - {m}{marker}")
                print()
            except Exception as exc:
                print(f"\n[Local Ollama Error]: {exc}\n")
            continue

        if cmd.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print(f"\nNebula AI: Current model is '{current_model}'. Usage: /model <name>\n")
                continue
            target_model = parts[1].strip()
            try:
                installed = list_installed_models()
                if installed and target_model not in installed:
                    print(
                        f"\nNebula AI: '{target_model}' is not installed locally yet. "
                        f"Run `ollama pull {target_model}` in PowerShell first.\n"
                    )
                    continue
                current_model = target_model
                append_log(f"SYSTEM: Switched model to {current_model}")
                print(f"\nNebula AI: Switched active model to '{current_model}'.\n")
            except Exception as exc:
                print(f"\n[Local Ollama Error]: {exc}\n")
            continue

        if cmd.startswith("calc "):
            result = calculate_expression(user_input[5:].strip())
            append_log(f"You: {user_input}")
            append_log(f"Nebula AI [calc]: {result}")
            print(f"\nNebula AI: {result}\n")
            continue

        messages.append({"role": "user", "content": user_input})
        append_log(f"You: {user_input}")
        print("\nNebula AI: ", end="", flush=True)
        try:
            reply = stream_ollama(messages, current_model)
            if reply:
                messages.append({"role": "assistant", "content": reply})
                append_log(f"Nebula AI [{current_model}]: {reply}")
        except Exception as exc:
            print(f"\n[Local Ollama Error]: {exc}\n")


if __name__ == "__main__":
    run_nebula()