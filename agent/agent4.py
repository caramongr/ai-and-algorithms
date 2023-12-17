class ModelBasedReflexAgent:
    def __init__(self, transition_model, sensor_model, rules):
        self.state = None  # Current conception of the world state
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.rules = rules
        self.action = None  # Most recent action

    def update_state(self, action, percept):

        print(f"Current state: {self.state}")
        print(f"Action: {action}")
        print(f"Percept: {percept}")

        # Update the agent's current state based on the transition model, action, and percept
        self.state = self.transition_model(self.state, action, percept)

        print(f"New state: {self.state}")

    def rule_match(self, state, rules):

        print(f"Matching rule for state: {state}")
        print(f"Available rules: {rules}")
        # Find a rule that matches the current state
        for rule in rules:
            if rule['condition'] == state:
                return rule
        return None

    def model_based_reflex_agent(self, percept):
        # Update the agent's internal state
        print(f"Current percept: {percept}")
        self.update_state(self.action, percept)
        
        
        # Match current state to a rule
        rule = self.rule_match(self.state, self.rules)

        print(f"Matched rule: {rule}")

        if rule:
            action = rule['action']
        else:
            action = 'NoOp'  # Default action if no rule matches

        self.action = action  # Update most recent action

        print(f"Action: {action}")
        return action

# Example transition and sensor models
def simple_transition_model(current_state, action, percept):
    # Placeholder transition model - update state based on action and percept
    return percept

def simple_sensor_model(world_state):
    # Placeholder sensor model - reflects world state in agent's percepts
    return world_state

# Example usage:
rules = [
    {'condition': 'Dirty', 'action': 'Suck'},
    {'condition': 'LocationA', 'action': 'Right'},
    {'condition': 'LocationB', 'action': 'Left'},
    # Add more rules as needed
]

agent = ModelBasedReflexAgent(simple_transition_model, simple_sensor_model, rules)

# Example percepts
percept1 = 'Dirty'
percept2 = 'LocationA'
percept3 = 'LocationB'

# Get actions for each percept
action1 = agent.model_based_reflex_agent(percept1)
action2 = agent.model_based_reflex_agent(percept2)
action3 = agent.model_based_reflex_agent(percept3)

# Display actions
print(f"Percept: {percept1}, Action: {action1}")
print(f"Percept: {percept2}, Action: {action2}")
print(f"Percept: {percept3}, Action: {action3}")
