"""Smallest possible M3 consumer: loads frozen item records and prints what
a compatibility/recommendation model would actually need. No model logic here
-- just proof that a consumer can read the contract without guessing fields.

Usage: python scripts/m3_consumer_example.py
"""

import json
from pathlib import Path

sample_path = (
    Path(__file__).resolve().parents[1] / "data/samples/items_representation/examples.json"
)
with sample_path.open(encoding="utf-8") as stream:
    records = json.load(stream)["valid_examples"]

for r in records:
    cat = r["category"]["common_category"] or f"(unmapped: {r['category']['source_label']})"
    has_color = r["color"]["palette"] is not None
    has_embedding = r["embedding"]["vector"] is not None
    print(
        f"{r['item_id']:45} split={r['split']['target_split']:10} "
        f"category={cat:12} color={'yes' if has_color else 'no':3} "
        f"embedding={'yes' if has_embedding else 'no'}"
    )
