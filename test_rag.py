"""Verify the updated app.py chunking logic works correctly."""
import re, os, sys
sys.path.insert(0, '.')

with open('recipes_data.txt', encoding='utf-8') as f:
    raw = f.read()

# app.py's new chunking logic
recipe_blocks = [b.strip() for b in re.split(r'\n---+\n', raw) if b.strip() and 'RECIPE:' in b]
print(f'Recipe chunks: {len(recipe_blocks)}')
for i, b in enumerate(recipe_blocks):
    m = re.search(r'^RECIPE:\s*(.+)', b, re.MULTILINE)
    name = m.group(1) if m else 'UNKNOWN'
    print(f'  {i+1}. {name} ({len(b)} chars)')

# Now parse the Chocolate Lava Cake block directly
target = [b for b in recipe_blocks if 'Chocolate Lava Cake' in b][0]

def parse(text):
    def field(p, d=''):
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else d
    def blk(p, d=''):
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else d
    name = field(r'^RECIPE:\s*(.+)')
    category = field(r'^CATEGORY:\s*(.+)')
    prep = field(r'^PREP TIME:\s*(.+)')
    ing_block = blk(r'INGREDIENTS:\s*\n(.*?)(?=\nINSTRUCTIONS:)')
    ingredients = [l.lstrip('- ').strip() for l in ing_block.splitlines() if l.strip().startswith('-')]
    ins_block = blk(r'INSTRUCTIONS:\s*\n(.*?)(?=\nTIPS:|\nNUTRITION:|$)')
    instructions = [l.strip() for l in ins_block.splitlines() if re.match(r'^\d+\.', l.strip())]
    tips = field(r'^TIPS:\s*(.+)')
    nutrition = field(r'^NUTRITION:\s*(.+)')
    return dict(name=name, category=category, prep=prep,
                ingredients=ingredients, instructions=instructions,
                tips=tips, nutrition=nutrition)

r = parse(target)
print()
print('=== Parsed Chocolate Lava Cake ===')
print('Name:', r['name'])
print('Category:', r['category'])
print('Prep:', r['prep'])
print('Ingredients:', len(r['ingredients']))
for i in r['ingredients'][:4]:
    print(' -', i)
print('Instructions:', len(r['instructions']))
for s in r['instructions'][:3]:
    print(' ', s)
print('Tips:', r['tips'][:80])
print('Nutrition:', r['nutrition'])
