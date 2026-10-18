"""Smallest possible M3 consumer: loads frozen item records and prints what
a compatibility/recommendation model would actually need. No model logic here
-- just proof that a consumer can read the contract without guessing fields.

Usage: python scripts/m3_consumer_example.py
"""
import json

records = json.load(open("data/samples/item_representation/examples.json", encoding="utf-8"))["valid_examples"]

for r in records:
    cat = r["category"]["common_category"] or f"(unmapped: {r['category']['source_label']})"
    has_color = r["color"]["palette"] is not None
    has_embedding = r["embedding"]["vector"] is not None
    print(f"{r['item_id']:45} split={r['split']['target_split']:10} "
          f"category={cat:12} color={'yes' if has_color else 'no':3} embedding={'yes' if has_embedding else 'no'}")
