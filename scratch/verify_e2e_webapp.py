"""Full-fidelity end-to-end verification script for the local web application."""

import re
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from app import create_app
from app.services.prediction_service import SYNTHETIC_DEMO_SEQUENCE

app = create_app()
client = app.test_client()

# 1. GET static css
r = client.get("/static/css/style.css")
assert r.status_code == 200, f"CSS failed: {r.status_code}"
assert "--bg-darkest" in r.get_data(as_text=True)
print("1. Static CSS delivered successfully (length: %d bytes)" % len(r.data))

# 2. GET static js
r = client.get("/static/js/app.js")
assert r.status_code == 200, f"JS failed: {r.status_code}"
assert "DOMContentLoaded" in r.get_data(as_text=True)
print("2. Static JS delivered successfully (length: %d bytes)" % len(r.data))

# 3. GET / (Home)
r = client.get("/")
assert r.status_code == 200
html = r.get_data(as_text=True)
assert "DNA Sequence Classification" in html
assert "csrf_token" in html
print("3. Home page verified with CSRF token and layout")

# 4. Form submit with CSRF token extraction
match = re.search(r'name="csrf_token"\s+value="([a-f0-9]+)"', html)
assert match, "CSRF token not found in form HTML"
token = match.group(1)
print(f"4. Extracted valid session CSRF token: {token[:12]}...")

# POST prediction with token
r = client.post(
    "/predict",
    data={"csrf_token": token, "sequence_text": SYNTHETIC_DEMO_SEQUENCE},
)
assert r.status_code == 200, f"Prediction failed: {r.status_code}"
res_html = r.get_data(as_text=True)
assert "Classification Results" in res_html
assert "Colorectal Cancer" in res_html
assert "Raw Decision Hyperplane Scores" in res_html
assert "Mandatory Educational Disclosure" in res_html
print("5. Prediction result verified with class, scores, and disclosure")

# 5. Check History page
r = client.get("/history")
assert r.status_code == 200
hist_html = r.get_data(as_text=True)
assert "Colorectal Cancer" in hist_html
print("6. Prediction history verified with recorded inference")

# 6. Check About page
r = client.get("/about")
assert r.status_code == 200
about_html = r.get_data(as_text=True)
assert "91.25%" in about_html
assert "0.9105" in about_html
assert "LinearSVC" in about_html
assert "Cervical recall" in about_html
print("7. About page verified with verified holdout benchmark results")

print("\n>>> ALL FULL-FIDELITY END-TO-END VERIFICATIONS PASSED SUCCESSFULLY! <<<")

