# imports
import mytextgrid
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_tgs_manual():
    tgs = []
    for f in sorted(listdir('force_aligned/manual')):
        if isfile(join('force_aligned/manual', f)):
            tgs.append(mytextgrid.read_from_file(join('force_aligned/manual', f)))

    df = pd.DataFrame(tgs)

    return tgs
