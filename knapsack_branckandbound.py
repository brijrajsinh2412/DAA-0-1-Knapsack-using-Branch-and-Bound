import plotly.graph_objects as go
import networkx as nx
from tabulate import tabulate

class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.ratio = value / weight

class Node:
    def __init__(self, level, value, weight, bound, items_taken):
        self.level = level
        self.value = value
        self.weight = weight
        self.bound = bound
        self.items_taken = items_taken

def bound(node, n, max_weight, items):
    if node.weight >= max_weight:
        return node.value
    if node.level + 1 < n:
        next_item = items[node.level + 1]
        remaining_weight = max_weight - node.weight
        return node.value + remaining_weight * next_item.ratio
    return node.value

def knapsack_branch_and_bound(max_weight, items):
    n = len(items)
    priority_queue = []
    root = Node(-1, 0, 0, 0, [])
    root.bound = bound(root, n, max_weight, items)
    priority_queue.append(root)

    max_profit = 0
    best_set = []
    tree = nx.DiGraph()
    node_id = 0
    tree.add_node(node_id, label=f"Root\nValue: 0.0, Weight: 0.0\nBound: {root.bound:.2f}")
    node_map = {tuple(root.items_taken): node_id}

    while priority_queue:
        priority_queue.sort(key=lambda x: x.bound, reverse=True)
        current_node = priority_queue.pop(0)
        current_id = node_map[tuple(current_node.items_taken)]

        if current_node.bound > max_profit:
            level = current_node.level + 1
            if level < n:
                # Include the item
                right_child = Node(level, current_node.value + items[level].value, current_node.weight + items[level].weight, 0, current_node.items_taken[:])
                right_child.items_taken.append(level)

                if right_child.weight <= max_weight and right_child.value > max_profit:
                    max_profit = right_child.value
                    best_set = right_child.items_taken[:]

                right_child.bound = bound(right_child, n, max_weight, items)
                node_id += 1
                tree.add_node(node_id, label=f"Include {level+1}\nValue: {right_child.value:.1f}, Weight: {right_child.weight:.1f}\nBound: {right_child.bound:.2f}")
                tree.add_edge(current_id, node_id)
                node_map[tuple(right_child.items_taken)] = node_id
                priority_queue.append(right_child)

                # Exclude the item
                left_child = Node(level, current_node.value, current_node.weight, 0, current_node.items_taken[:])
                left_child.bound = bound(left_child, n, max_weight, items)
                node_id += 1
                tree.add_node(node_id, label=f"Exclude {level+1}\nValue: {left_child.value:.1f}, Weight: {left_child.weight:.1f}\nBound: {left_child.bound:.2f}")
                tree.add_edge(current_id, node_id)
                node_map[tuple(left_child.items_taken)] = node_id
                priority_queue.append(left_child)

    # Convert best_set to 0/1 format
    taken = [0] * n
    for index in best_set:
        taken[index] = 1

    return max_profit, taken, tree

def plot_tree_interactive(tree):
    # Manually create a hierarchical layout
    pos = hierarchy_pos(tree, root=0, width=1.0, vert_gap=0.2, vert_loc=0)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in tree.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=20,
            color=[],
            colorbar=dict(
                thickness=15,
                title='Node Depth',
                xanchor='left',
                yanchor='middle'
            ),
            line_width=2))

    for node in tree.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([f"{tree.nodes[node]['label']}"])
        node_trace['marker']['color'] += tuple([len(list(nx.shortest_path_length(tree, source=node).keys()))])

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Branch and Bound Tree",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=1.1,
                    font=dict(size=16, color='black')
                )],
                title='Branch and Bound Tree Visualization',
                template='plotly_dark'
                ))
    fig.show()

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    G: the graph

    root: the root node of current branch
    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    def h_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        """
    
        pos: a dict saying where all nodes go if they have been assigned
        parent: a node in the graph that is the parent of the root node
        """
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            next_x = xcenter - width / 2 - dx / 2
            for child in children:
                next_x += dx
                pos = h_pos(G, child, width=dx, vert_gap=vert_gap,
                            vert_loc=vert_loc - vert_gap, xcenter=next_x, pos=pos, parent=root)
        return pos

    return h_pos(G, root, width, vert_gap, vert_loc, xcenter)

def main():
    try:
        # Static values (commented out)
        use_static_values = True

        if use_static_values:
            n = 5
            items = [
                Item(40, 1),
                Item(30, 2),
                Item(25, 3),
                Item(15, 5),
                Item(12, 4),
            ]
            max_weight = 10
        else:
            # User input
            n = int(input("Enter the number of items: "))
            items = []
            print("Enter the value and weight of each item:")
            for i in range(n):
                value = float(input(f"Item {i + 1} - Value: "))
                weight = float(input(f"Item {i + 1} - Weight: "))
                items.append(Item(value, weight))

            max_weight = float(input("Enter the maximum weight capacity of the knapsack: "))

        # Sort items by value-to-weight ratio in descending order
        items.sort(key=lambda x: x.ratio, reverse=True)

        max_profit, taken, tree = knapsack_branch_and_bound(max_weight, items)

        # Display items in a table format with value-to-weight ratio
        table = [[i + 1, f"{items[i].value:.1f}", f"{items[i].weight:.1f}", f"{items[i].ratio:.2f}"] for i in range(n)]
        print("\nItems Table:")
        print(tabulate(table, headers=["Item", "Value", "Weight", "Value/Weight"], tablefmt="grid"))

        print("\nMaximum profit:", max_profit)

        # Display items taken in 0/1 format
        print("\nItems taken in the knapsack (0/1 format):")
        print(taken)

        # Plot tree graph
        plot_tree_interactive(tree)

    except ValueError as e:
        print(f"Error: {e}. Please enter valid numerical inputs.")

if __name__ == "__main__":
    main()
