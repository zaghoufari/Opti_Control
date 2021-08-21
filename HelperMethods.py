# HelperMethods

import numpy as np

def Distinct(arr):
    seen = []
    for m in arr:
        if len(seen) == 0:
            seen.append(m)
            continue

        for aleadySeen in seen:
            if not np.array_equal(m, aleadySeen):
                seen.append(m)
    return seen


def PrintLineCsv(f, vals):
    for i in range(len(vals)):
        f.write(str(float(vals[i])))
        if i < len(vals) - 1:
            f.write(",")



