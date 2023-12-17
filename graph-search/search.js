class Node {
    constructor(name) {
        this.name = name;
        this.children = [];
    }
}

class GraphProblem {
    constructor() {
        // Create a simple graph for demonstration purposes
        console.log("Creating a graph problem");
        this.graph = {};
        this.graph['A'] = ['B', 'C'];
        this.graph['B'] = ['D', 'E'];
        this.graph['C'] = ['F'];
        this.graph['D'] = [];
        this.graph['E'] = ['F'];
        this.graph['F'] = [];
    }

    initialState() {
        return 'A';
    }

    isGoal(state) {
        return state === 'F';
    }

    actions(state) {
        return this.graph[state];
    }

    result(state, action) {
        return action;
    }

    actionCost() {
        // For this example, we assume all actions have a cost of 1
        return 1;
    }
}

function bestFirstSearch(problem, f) {

console.log("problem"+JSON.stringify(problem));
console.log("f"+f);

    const node = new Node(problem.initialState());
    const frontier = [];
    const reached = {};

    frontier.push([f(node), node]);

    console.log("frontier"+JSON.stringify(frontier));


    reached[problem.initialState()] = node;

    console.log("reached"+JSON.stringify(reached));

    while (frontier.length > 0) {
        frontier.sort((a, b) => a[0] - b[0]); // Sort based on priority (f value)

        console.log("frontier prin shift"+JSON.stringify(frontier));
        const [_, currentNode] = frontier.shift();
        console.log("frontier meta shift"+JSON.stringify(frontier));
        console.log("currentNode"+JSON.stringify(currentNode));

        if (problem.isGoal(currentNode.name)) {
            return currentNode;
        }

        for (const childName of problem.actions(currentNode.name)) {
            const childNode = new Node(childName);
            currentNode.children.push(childNode);

            if (!(childName in reached)) {
                reached[childName] = childNode;
                frontier.push([f(childNode), childNode]);
            }
        }
    }

    return null;
}

// Heuristic function for the example problem (distance to goal)
function heuristic(node) {
    // For simplicity, this heuristic assumes straight-line distance
    // between nodes (which may not always be accurate in real problems)
    const distances = {
        'A': 3,
        'B': 2,
        'C': 2,
        'D': 1,
        'E': 1,
        'F': 0
    };
    return distances[node.name];
}

// Call the search function
const problem = new GraphProblem();
const solution = bestFirstSearch(problem, heuristic);

if (solution) {
    // If a solution is found, build and display the path
    const buildPath = (node) => {
        let path = node.name;
        while (node.parent) {
            node = node.parent;
            path = node.name + ' -> ' + path;
        }
        return path;
    };

    console.log("Solution found. Path:", buildPath(solution));
} else {
    console.log("No solution found");
}