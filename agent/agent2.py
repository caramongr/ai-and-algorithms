def reflex_vacuum_agent(location_status):
    location, status = location_status

    if status == "Dirty":
        return "Suck"
    elif location == "A":
        return "Right"
    elif location == "B":
        return "Left"

# Example usage:
location_status_A = ("A", "Dirty")
location_status_B = ("B", "Clean")

action_A = reflex_vacuum_agent(location_status_A)
action_B = reflex_vacuum_agent(location_status_B)

print(f"Action for Location A: {action_A}")
print(f"Action for Location B: {action_B}")
