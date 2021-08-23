import re
import os
import sys
import random
import numpy as np
from pathlib import Path

ImageSuffixes = set([
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'
])


class Splitter:
    def __init__(self, base_dir='Sketchy'):
        self.base_dir = base_dir
        self.sketch_re = re.compile( r"^(.*)-\d+\.png$" )

    def sketch_to_photo_path(self, sketch_path, photo_dir):
        m = self.sketch_re.match(sketch_path.name)
        return photo_dir/f"{m.group(1)}.jpg"

    def split(self, train_split, test_split, valid_split):
        ratios = np.array([train_split, test_split, valid_split], dtype=float)
        ratios /= np.sum(ratios)

        paths = dict()
        paths['base'] = Path(self.base_dir)
        paths['source'] = paths['base'] / 'all'
        for phase in ['train', 'test', 'valid']:
            paths[phase] = paths['base'] / phase
            if not paths[phase].exists():
                paths[phase].mkdir()

        paths['source_sketch'] = paths['source'] / 'sketch'
        paths['source_photo'] = paths['source'] / 'photo'
        cat_dirs = [(path, paths['source_photo']/path.name) for path in paths['source_sketch'].iterdir() if path.is_dir()]

        for sketch_cat_dir, photo_cat_dir in cat_dirs:
            cat_pairs = set([
                (path, self.sketch_to_photo_path(path, photo_cat_dir))
                for path in sketch_cat_dir.iterdir()
                if path.suffix in ImageSuffixes
            ])
            cat_name = sketch_cat_dir.name

            size = dict()
            # Calculate number of image files in each split
            size['train'], size['test'], size['valid'] = (ratios * len(cat_pairs)).astype(int)
            # Total of splits should equal total number of images.
            # If rounding causes otherwise, adjust 'train' set to fit.
            size['train'] += len(cat_pairs) - sum(size.values())

            print( f"Category: {cat_name}, Total pictures: {len(cat_pairs)}" )
            print( f"\tSplits: Train({size['train']}), Test({size['test']}), Validation({size['valid']})" )

            for phase in ['train', 'test', 'valid']:
                sample = random.sample(list(cat_pairs), size[phase])

                target_paths = dict()
                for img_type in ('sketch', 'photo'):
                    target_paths[img_type] = paths[phase] / img_type / cat_name
                    target_paths[img_type].mkdir(parents=True, exist_ok=True)

                for sk_ph_paths in sample:
                    for img_type, file_path in zip( ('sketch','photo'), sk_ph_paths ):
                        target_path = target_paths[img_type] / file_path.name
                        rel_file_path = Path( '..', '..', '..', *file_path.parts[1:] )
                        if not target_path.exists():
                            target_path.symlink_to(rel_file_path)

                cat_pairs -= set(sample)

if __name__ == '__main__':
    splitter = Splitter('Sketchy')
    splitter.split(95, 5, 0)
