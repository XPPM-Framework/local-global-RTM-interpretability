import os
from pathlib import Path

MY_WORKSPACE_DIR = Path(os.getenv("MY_WORKSPACE_DIR", "."))
print(f"Using workspace directory: {MY_WORKSPACE_DIR.resolve()}")
