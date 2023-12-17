class LookupAgent:
    def __init__(self, table):
        self.percepts = []
        self.table = table

    def get_action(self, percept):
        self.percepts.append(percept)
        action = self.lookup(self.percepts, self.table)
        return action

    def lookup(self, percepts, table):
        key = tuple(percepts)
        print("key is: ", key)
        return table.get(key, 'NoOp')  # 'NoOp' if no matching action is found

# Example table for the AI agent
ai_table = {
    (0,): 'ACTION_A',
    (1,): 'ACTION_B',
    (0, 1): 'ACTION_C',
    (1, 0): 'ACTION_D',
}

# Example of using the LookupAgent
ai_agent = LookupAgent(ai_table)

# Simulate the agent's actions based on percepts
percepts_sequence = [0, 1, 1, 0, 1, 0]  # Example percept sequence
for percept in percepts_sequence:
    action = ai_agent.get_action(percept)
    print(f"Percept: {percept}, Action: {action}")