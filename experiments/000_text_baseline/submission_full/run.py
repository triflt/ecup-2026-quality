import argparse
import csv
import re
from pathlib import Path

from src.model import StrongTextBundle, compose_text
from src.output import format_result


FORMAT = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_data_path", "--test-data-path", dest="input", required=True)
    parser.add_argument("-o", "--output_path", "--output-path", dest="output", required=True)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "name", "description", "category"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Input misses columns: {sorted(required - set(rows[0]))}")
    names = [row.get("name", "") for row in rows]
    descriptions = [row.get("description", "") for row in rows]
    categories = [row.get("category", "") for row in rows]
    texts = [compose_text(a, b) for a, b in zip(names, descriptions)]
    bundle = StrongTextBundle.load(str(Path(__file__).resolve().parent / "strong_text.joblib"))
    _, predictions = bundle.predict(names, texts, categories)
    output_rows = []
    for row, prediction in zip(rows, predictions):
        result = format_result(row["category"], prediction, row.get("name", ""), row.get("description", ""))
        if FORMAT.fullmatch(result) is None:
            raise ValueError(f"Invalid output: {result!r}")
        output_rows.append({"id": row["id"], "result": result})
    with open(args.output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "result"])
        writer.writeheader()
        writer.writerows(output_rows)
    if len(output_rows) != len(rows):
        raise RuntimeError("Output row count mismatch")
    print(f"saved rows={len(output_rows)} path={args.output}")


if __name__ == "__main__":
    main()
