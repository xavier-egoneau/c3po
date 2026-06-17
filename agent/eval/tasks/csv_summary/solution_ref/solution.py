import csv
import json
from collections import defaultdict


def main():
    totals = defaultdict(float)
    with open("input.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            totals[row["category"]] += float(row["amount"])
    summary = {category: totals[category] for category in sorted(totals)}
    with open("summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
