import sys
from agent.core import process_stream, handle_command

tests = [
    ("/help", "Comandos"),
    ("!echo hello", "hello"),
    ("read: app.py", "app.py"),
    ("glob: *.py", "Encontrados"),
    ("ls", "app.py"),
]

for msg, expected in tests:
    if msg.startswith("/"):
        r = handle_command(msg)
    else:
        r = "".join(process_stream(msg))
    status = "OK" if expected in r else "FAIL"
    print(f"[{status}] {msg}")
    print(f"  -> {r[:80]}")
print("ALL DONE")
