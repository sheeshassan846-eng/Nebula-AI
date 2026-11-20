import ast
import operator
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

MODEL_NAME = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = (
    "You are Nebula AI, an intelligent, structured, and reliable assistant. "
    "Use Google Search when fresh or verifiable real-world facts are needed, "
    "and use calculate_expression for exact math instead of guessing."
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


def _eval_ast_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_ast_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _SAFE_OPERATORS[type(node.op)](left, right)
    raise ValueError("Unsupported mathematical expression.")


def calculate_expression(expression: str) -> str:
    """Safely evaluates a basic arithmetic expression such as '(145 * 8) / 2'.

    Args:
        expression: A mathematical expression using numbers and +, -, *, /, **, %.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_ast_node(parsed)
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


def create_chat_session() -> genai.chats.Chat:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: GEMINI_API_KEY is missing. Add it to your .env file before running Nebula AI.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
        tools=[
            types.Tool(google_search=types.GoogleSearch()),
            calculate_expression,
        ],
    )
    return client.chats.create(model=MODEL_NAME, config=config)


def run_nebula() -> None:
    chat = create_chat_session()
    print(f"Nebula AI ({MODEL_NAME}) is online with Search + Tools. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nNebula AI: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Nebula AI: Goodbye!")
            break

        try:
            print("\nNebula AI: ", end="", flush=True)
            got_text = False
            for chunk in chat.send_message_stream(user_input):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    got_text = True
            if not got_text:
                print("(No text response returned.)", end="")
            print("\n")
        except APIError as exc:
            print(f"\n[API Error {exc.code}]: {exc.message}\n", file=sys.stderr)
        except Exception as exc:
            print(f"\n[Unexpected Error]: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    run_nebula()
