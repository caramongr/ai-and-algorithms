class SimpleReflexAgent:
    def __init__(self, rules):
        self.rules = rules

    def interpret_input(self, percept):
        # Placeholder for input interpretation logic
        # You may customize this part based on your specific scenario
        return percept

    def rule_match(self, state, rules):
        for rule in rules:
            if rule['condition'] == state:
                return rule
        return None

    def simple_reflex_agent(self, percept):
        print(f"Current percept: {percept}")
        state = self.interpret_input(percept)
        print(f"Current state: {state}")
        rule = self.rule_match(state, self.rules)
        print(f"Matched rule: {rule}")

        if rule:
            action = rule['action']
        else:
            action = 'NoOp'  # Default action if no rule matches

        return action

# Example usage:
rules = [
    {'condition': 'Dirty', 'action': 'Suck'},
    {'condition': 'LocationA', 'action': 'Right'},
    {'condition': 'LocationB', 'action': 'Left'},
]

simple_agent = SimpleReflexAgent(rules)

# Example percepts
percept1 = {'location': 'A', 'status': 'Dirty'}
percept2 = {'location': 'B', 'status': 'Clean'}
percept3 = {'location': 'C', 'status': 'Dirty'}  # Unknown location

# Get actions for each percept
action1 = simple_agent.simple_reflex_agent(percept1)
action2 = simple_agent.simple_reflex_agent(percept2)
action3 = simple_agent.simple_reflex_agent(percept3)

# Display actions
print(f"Percept: {percept1}, Action: {action1}")
print(f"Percept: {percept2}, Action: {action2}")
print(f"Percept: {percept3}, Action: {action3}")
