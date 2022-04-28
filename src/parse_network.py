
def parse_network(casefile):
    # read data from file
    # apply Kron reduction?
    gens = []
    ibrs = []
    branches = []
    # assign indexes
    for ele in branches:
        ele.assign_indexes(gens, ibrs)
    return gens, ibrs, branches