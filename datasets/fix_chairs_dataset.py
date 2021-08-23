import os
from pathlib import Path

fixes = (
    ('Chairs/trainA/sdich070-laque-rouge_A1YX741QQSMKPS.png', 'sdich070-laque-rouge_2.png'),
    ('Chairs/trainA/sdich070-laque-rouge_A2NWM33YRH533Q.png', 'sdich070-laque-rouge_3.png'),
    ('Chairs/trainA/sdich070-laque-rouge_AI6TD8PM938FQ.png', 'sdich070-laque-rouge_4.png')
)

for old_fpath, new_fname in fixes:

    old_fpath = Path(old_fpath)
    new_fpath = old_fpath.with_name(new_fname)

    if not old_fpath.exists():
        print( f"Warning: {old_fpath} does not exist. Skipping" )
        continue
    if new_fpath.exists():
        print( f"Warning: {new_fpath} already exists. Skipping" )
        continue

    print( f"Renaming {old_fpath} -> {new_fpath}" )
    old_fpath.rename(new_fpath)
