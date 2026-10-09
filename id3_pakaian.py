import math
from collections import Counter

DATA = [
    {"Warna": "Merah",  "Ukuran": "S", "Bahan": "Katun", "Kategori": "Casual"},
    {"Warna": "Biru",   "Ukuran": "M", "Bahan": "Sutra", "Kategori": "Formal"},
    {"Warna": "Hijau",  "Ukuran": "L", "Bahan": "Wol",   "Kategori": "Casual"},
    {"Warna": "Kuning", "Ukuran": "S", "Bahan": "Sutra", "Kategori": "Casual"},
    {"Warna": "Merah",  "Ukuran": "M", "Bahan": "Katun", "Kategori": "Casual"},
    {"Warna": "Biru",   "Ukuran": "L", "Bahan": "Wol",   "Kategori": "Formal"},
    {"Warna": "Hijau",  "Ukuran": "S", "Bahan": "Katun", "Kategori": "Casual"},
    {"Warna": "Kuning", "Ukuran": "M", "Bahan": "Sutra", "Kategori": "Formal"},
]

TARGET = "Kategori"
ATTRS = ["Warna", "Ukuran", "Bahan"]


def entropy(rows):
    counts = Counter(row[TARGET] for row in rows)
    total = len(rows)
    result = 0.0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)
    return result


def information_gain(rows, attribute):
    base = entropy(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row[attribute], []).append(row)

    remainder = 0.0
    for subset in groups.values():
        remainder += (len(subset) / len(rows)) * entropy(subset)
    return base - remainder


def majority_class(rows):
    return Counter(row[TARGET] for row in rows).most_common(1)[0][0]


def id3(rows, attributes):
    labels = [row[TARGET] for row in rows]

    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return majority_class(rows)

    best = max(attributes, key=lambda a: information_gain(rows, a))
    tree = {best: {}, "_default": majority_class(rows)}

    values = sorted(set(row[best] for row in rows))
    remaining = [a for a in attributes if a != best]

    for value in values:
        subset = [row for row in rows if row[best] == value]
        tree[best][value] = id3(subset, remaining)

    return tree


def predict(tree, sample):
    if isinstance(tree, str):
        return tree

    attribute = next(k for k in tree.keys() if k != "_default")
    value = sample.get(attribute)
    branch = tree[attribute].get(value)

    if branch is None:
        return tree["_default"]
    return predict(branch, sample)


print("Entropy awal:", round(entropy(DATA), 6))
for attr in ATTRS:
    print(f"Gain {attr}:", round(information_gain(DATA, attr), 6))

TREE = id3(DATA, ATTRS)
print("Pohon:", TREE)

correct = 0
for i, row in enumerate(DATA, start=1):
    pred = predict(TREE, row)
    correct += pred == row[TARGET]
    print(i, row["Warna"], row["Ukuran"], row["Bahan"], "->", pred)

print(f"Akurasi data latih: {correct}/{len(DATA)} = {correct/len(DATA)*100:.0f}%")