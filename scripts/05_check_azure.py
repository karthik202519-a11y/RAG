"""Check .env loads and one embedding call succeeds. Prints no secrets."""
import os

from azure_client import REQUIRED, get_client

for k in REQUIRED:
    v = os.getenv(k)
    print(f"{k}: {'set' if v else 'MISSING'}" + (f" ({v})" if v and k != "AZURE_OPENAI_API_KEY" else ""))

client, deployment = get_client()
r = client.embeddings.create(model=deployment, input=["What approval is required before deployment?"])
vec = r.data[0].embedding
print(f"OK: vector length {len(vec)}, tokens used {r.usage.total_tokens}")
