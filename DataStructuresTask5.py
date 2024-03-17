# Define two sets
E = {0, 2, 4, 6, 8}
N = {1, 2, 3, 4, 5}

# Union of E and N
union_EN = E.union(N)
print("Union of E and N is", union_EN)

# Intersection of E and N
intersection_EN = E.intersection(N)
print("Intersection of E and N is", intersection_EN)

# Difference of E and N (elements in E but not in N)
difference_EN = E.difference(N)
print("Difference of E and N is", difference_EN)

# Symmetric difference of E and N (elements in either E or N but not in both)
symmetric_difference_EN = E.symmetric_difference(N)
print("Symmetric difference of E and N is", symmetric_difference_EN)
