class GameState {
    constructor(total = 0, playerTurn = 1) {
        this.total = total;
        this.playerTurn = playerTurn;
    }

    getPossibleMoves() {
        if (this.total >= 10) {
            return [];
        }
        return [1, 2];
    }

    isTerminal() {
        return this.total >= 10;
    }

    makeMove(move) {
        return new GameState(this.total + move, -this.playerTurn);
    }
}

class MCTSNode {
    constructor(gameState, parent = null, move = null) {
        this.gameState = gameState;
        this.parent = parent;
        this.move = move;
        this.children = [];
        this.wins = 0;
        this.visits = 0;
    }

    selectChild() {
        if (this.children.length === 0) {
            return null;
        }

        return this.children.reduce((maxChild, child) => {
            const ucb1 = (child.wins / child.visits) + Math.sqrt(2) * Math.sqrt(Math.log(this.visits) / child.visits);
            return ucb1 > maxChild.ucb1 ? { child, ucb1 } : maxChild;
        }, { child: null, ucb1: -Infinity }).child;
    }
}

function expand(node) {
    const moves = node.gameState.getPossibleMoves();
    moves.forEach(move => {
        const nextGameState = node.gameState.makeMove(move);
        const childNode = new MCTSNode(nextGameState, node, move);
        node.children.push(childNode);
    });
}

function simulate(gameState) {
    let currentState = gameState;
    while (!currentState.isTerminal()) {
        const possibleMoves = currentState.getPossibleMoves();
        const move = possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
        currentState = currentState.makeMove(move);
    }
    return currentState;
}

function backpropagate(node, winner) {
    while (node !== null) {
        node.visits += 1;
        if (node.gameState.playerTurn === winner) {
            node.wins += 1;
        }
        node = node.parent;
    }
}

function monteCarloTreeSearch(root, iterations = 1000) {
    for (let i = 0; i < iterations; i++) {
        let node = root;

        // Selection
        while (node && node.children.length > 0) {
            const selectedChild = node.selectChild();
            if (selectedChild) {
                node = selectedChild;
            } else {
                // Handle the case where selectChild returns null
                break;
            }
        }

        if (!node) {
            continue; // Skip this iteration if node is null
        }

        // Expansion
        if (!node.gameState.isTerminal() && node.children.length === 0) {
            expand(node);
        }

        // Simulation
        const winner = simulate(node.gameState).playerTurn;

        // Backpropagation
        backpropagate(node, winner);
    }

    return root.children.reduce((maxChild, child) => {
        return child.visits > maxChild.visits ? child : maxChild;
    }, root.children[0]).move;
}

// Example usage
const root = new MCTSNode(new GameState());
const bestMove = monteCarloTreeSearch(root);
console.log(`The best move is: ${bestMove}`);
