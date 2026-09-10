import re

# Read the raw file to test
with open('recipes_data.txt', encoding='utf-8') as f:
    raw = f.read()

# Simulate what FAISS returns - split by ---
blocks = re.split(r'-{3,}', raw)

# Get the Chocolate Lava Cake block
target = [b for b in blocks if 'Chocolate Lava Cake' in b][0]

print("=== LINE ENDINGS ===")
print("Has \\r\\n:", '\r\n' in target)
print("Has \\r:", '\r' in target)
print("First 80 chars repr:", repr(target[:80]))
print()

# Test ingredient pattern
m1 = re.search(r'INGREDIENTS:\s*\n(.*?)(?=\nINSTRUCTIONS:)', target, re.DOTALL)
print("Pattern 1 (\\n):", bool(m1))

m2 = re.search(r'INGREDIENTS:\s*[\r\n]+(.*?)(?=[\r\n]+INSTRUCTIONS:)', target, re.DOTALL)
print("Pattern 2 ([\\r\\n]+):", bool(m2))
if m2:
    lines = [l.lstrip('- ').strip() for l in m2.group(1).splitlines() if l.strip().startswith('-')]
    print("Ingredients found:", len(lines))
    for l in lines[:3]:
        print(' -', l)

m3 = re.search(r'INSTRUCTIONS:\s*[\r\n]+(.*?)(?=[\r\n]+TIPS:|[\r\n]+NUTRITION:|$)', target, re.DOTALL)
print()
print("Instructions pattern:", bool(m3))
if m3:
    steps = [l.strip() for l in m3.group(1).splitlines() if re.match(r'^\d+\.', l.strip())]
    print("Steps found:", len(steps))
    for s in steps[:3]:
        print(' ', s)
