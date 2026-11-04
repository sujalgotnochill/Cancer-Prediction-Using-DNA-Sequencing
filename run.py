#!/usr/bin/env python
"""Local entry point for the DNA Sequence Classifier web application.

Starts a local Flask development server at http://127.0.0.1:5000.
Debug mode is disabled by default and external network exposure is blocked.
"""

import os
import sys
from pathlib import Path

# Add project root and src/ directory to sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    # Never run with debug=True by default in production or normal entry point
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")

    print("\n" + "=" * 65)
    print("  DNA Sequence Classification — Local Web Application")
    print("=" * 65)
    print(f"  Local URL:        http://{host}:{port}")
    print("  Binding:          Localhost only (isolated from external network)")
    print(f"  Debug Mode:       {debug}")
    print("  Model Candidate:  LinearSVC (k=4, TF-IDF standard representation)")
    print("  Dataset Notice:   Educational sequence-source classification only")
    print("=" * 65 + "\n")

    app.run(host=host, port=port, debug=debug)

