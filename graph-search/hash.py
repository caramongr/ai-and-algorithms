

start = {}
start["a"] = 6
start["b"] = 2

a = {}
a["fin"] = 4

b = {}
b["a"] = 3
b["fin"] = 5

fin = {}

graph = {}
graph["start"] = start
graph["a"] = a
graph["b"] = b
graph["fin"] = fin

import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_edge("start", "a", weight=6)
G.add_edge("start", "b", weight=2)
G.add_edge("a", "fin", weight=1)
G.add_edge("b", "a", weight=3)
G.add_edge("b", "fin", weight=5)

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.axis('off')
plt.show()



