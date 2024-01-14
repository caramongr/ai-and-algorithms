def ac3(csp):
    queue = csp.arcs()
    while queue:
        (xi, xj) = queue.pop(0)
        if revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False  # No valid values left, failure
            for xk in csp.neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True  # CSP is arc consistent

def revise(csp, xi, xj):
    revised = False
    for x in csp.domains[xi][:]:
        if not any([csp.constraints(xi, x, xj, y) for y in csp.domains[xj]]):
            csp.domains[xi].remove(x)
            revised = True
    return revised

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.neighbors = {var: [v for v in self.variables if v != var] for var in self.variables}

    def arcs(self):
        return [(x, y) for x in self.variables for y in self.neighbors[x]]

# Define the CSP
variables = ['A', 'B', 'C']
domains = {v: [1, 2, 3] for v in variables}

def constraints(x, vx, y, vy):
    if x == 'A' and y == 'B':
        return vx != vy
    if x == 'B' and y == 'C':
        return vx != vy
    if x == 'A' and y == 'C':
        return vx < vy
    return True  # Default case

# Initialize CSP
csp_instance = CSP(variables, domains, constraints)

# Run AC-3
ac3(csp_instance)

# Output results
print("Domains after AC-3:")
for v in variables:
    print(f"{v}: {csp_instance.domains[v]}")
