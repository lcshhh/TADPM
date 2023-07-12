import os
from pathlib import Path
lines = []
path = Path('/data/lcs/finetuned_teeth/merged_after')
for p in path.iterdir():
    name = p.name
    lines.append(int(name.split('.')[0]))
lines = sorted(lines)
with open('check2.txt','w',encoding='utf-8') as file:
    for line in lines:
        file.write(str(line)+'\n')

