import os
import re

with open("requirements.txt", "rb") as f:
    content = f.read()

try:
    text = content.decode('utf-8')
except Exception:
    text = content.decode('utf-16-le', errors='ignore')

text = text.replace('?', '').strip()

if 'dice-ml' not in text:
    text += "\ndice-ml\nshap\nmatplotlib\npytest\n"

with open("requirements.txt", "w", encoding='utf-8') as f:
    f.write(text)
print("requirements.txt fixed")
