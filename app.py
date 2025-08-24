from flask import Flask, render_template, request, jsonify
from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import io
import base64

app = Flask(__name__)

# AVL Tree Node class
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1

# AVL Tree functions
def height(node):
    if node is None:
        return 0
    return node.height

def max_val(a, b):
    return a if a > b else b

def newNode(data):
    return Node(data)

def getBalance(node):
    if node is None:
        return 0
    return height(node.left) - height(node.right)

def rightRotate(y):
    x = y.left
    T2 = x.right
    x.right = y
    y.left = T2
    y.height = max_val(height(y.left), height(y.right)) + 1
    x.height = max_val(height(x.left), height(x.right)) + 1
    return x

def leftRotate(x):
    y = x.right
    T2 = y.left
    y.left = x
    x.right = T2
    x.height = max_val(height(x.left), height(x.right)) + 1
    y.height = max_val(height(y.left), height(y.right)) + 1
    return y

def insert(node, data):
    if node is None:
        return newNode(data)
    if data < node.data:
        node.left = insert(node.left, data)
    elif data > node.data:
        node.right = insert(node.right, data)
    else:
        return node
    
    node.height = 1 + max_val(height(node.left), height(node.right))
    balance = getBalance(node)
    
    # Left-Left case
    if balance > 1 and data < node.left.data:
        return rightRotate(node)
    # Right-Right case
    if balance < -1 and data > node.right.data:
        return leftRotate(node)
    # Left-Right case
    if balance > 1 and data > node.left.data:
        node.left = leftRotate(node.left)
        return rightRotate(node)
    # Right-Left case
    if balance < -1 and data < node.right.data:
        node.right = rightRotate(node.right)
        return leftRotate(node)
    return node

def inOrder(root, ls):
    if root is not None:
        inOrder(root.left, ls)
        ls.append(root.data)
        inOrder(root.right, ls)

def preOrder(root, ls):
    if root is not None:
        ls.append(root.data)
        preOrder(root.left, ls)
        preOrder(root.right, ls)

def visualize_tree(node, graph=None):
    if graph is None:
        graph = Digraph()
        graph.attr('node', shape='circle')
    
    if node is not None:
        graph.node(str(node.data), str(node.data))
        if node.left:
            graph.edge(str(node.data), str(node.left.data))
            visualize_tree(node.left, graph)
        if node.right:
            graph.edge(str(node.data), str(node.right.data))
            visualize_tree(node.right, graph)
    return graph

# Heap Sort functions
def sift_up(arr, i, steps, sorted_indices):
    parent = (i - 1) // 2
    while i > 0 and arr[parent] < arr[i]:
        steps.append((arr.copy(), sorted_indices.copy(), i, "red"))
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent
        parent = (i - 1) // 2
    steps.append((arr.copy(), sorted_indices.copy(), i, "green"))

def topDownHeapSort(arr):
    n = len(arr)
    steps = [(arr.copy(), set(), -1, "")]
    sorted_indices = set()
    
    # Build heap using top-down approach
    for i in range(n):
        sift_up(arr, i, steps, sorted_indices)
    
    # Extract elements from heap
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        sorted_indices.add(i)
        steps.append((arr.copy(), sorted_indices.copy(), i, "green"))
        if i > 1:  # Only heapify if there are elements left to heapify
            heapify(arr, i, 0, steps, sorted_indices)
    
    return steps

def sift_down(arr, n, i, steps, sorted_indices):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        steps.append((arr.copy(), sorted_indices.copy(), i, "red", largest))
        arr[i], arr[largest] = arr[largest], arr[i]
        sift_down(arr, n, largest, steps, sorted_indices)
    else:
        # Add step even when no swap occurs to show the current state
        steps.append((arr.copy(), sorted_indices.copy(), i, "blue", -1))

def bottomUpHeapSort(arr):
    n = len(arr)
    steps = [(arr.copy(), set(), -1, "", -1)]
    sorted_indices = set()
    
    # Build heap using bottom-up approach
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, n, i, steps, sorted_indices)
    
    # Extract elements from heap
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        sorted_indices.add(i)
        steps.append((arr.copy(), sorted_indices.copy(), 0, "green", i))
        if i > 1:  # Only heapify if there are elements left to heapify
            sift_down(arr, i, 0, steps, sorted_indices)
    
    return steps

def heapify(arr, n, i, steps, sorted_indices):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        steps.append((arr.copy(), sorted_indices.copy(), i, "red"))
        heapify(arr, n, largest, steps, sorted_indices)

# Old heap sort functions removed - using improved versions above

# Graph Coloring functions
def is_valid_coloring(graph, coloring):
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            return False
    return True

def get_adjacent_colors(graph, node, coloring):
    adjacent_colors = set()
    for neighbor in graph.neighbors(node):
        if neighbor in coloring:
            adjacent_colors.add(coloring[neighbor])
    return adjacent_colors

def find_next_color(adjacent_colors):
    color = 0
    while color in adjacent_colors:
        color += 1
    return color

def greedy_coloring(graph):
    coloring = {}
    for node in graph.nodes():
        adjacent_colors = get_adjacent_colors(graph, node, coloring)
        color = find_next_color(adjacent_colors)
        coloring[node] = color
    return coloring

def generate_random_graph(n_nodes, edge_prob):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                G.add_edge(i, j)
    return G


def visualize_heap(arr, sorted_indices, current_index, current_color, swap_with=None):
    """Generate clean tree visualization using CSS positioning"""
    
    if not arr:
        return "<div>Empty heap</div>"
    
    n = len(arr)
    if n == 0:
        return "<div>Empty heap</div>"
    
    # Calculate tree height
    import math
    height = math.ceil(math.log2(n + 1))
    
    # Base dimensions
    node_size = 40
    level_height = 60
    base_width = (2 ** (height - 1)) * node_size * 2
    
    html = f"""
    <div style="position: relative; width: {base_width}px; height: {height * level_height + 50}px; margin: 20px auto; background: #f8f9fa; border-radius: 10px; padding: 20px;">
    """
    
    def get_node_style(index):
        """Get styling for node based on its state"""
        if index == current_index:
            return "background-color: #e74c3c; color: white; border: 3px solid #c0392b;"
        elif index == swap_with:
            return "background-color: #f39c12; color: white; border: 3px solid #e67e22;"
        elif index in sorted_indices:
            return "background-color: #27ae60; color: white; border: 3px solid #229954;"
        else:
            return "background-color: #3498db; color: white; border: 3px solid #2980b9;"
    
    def calculate_position(index):
        """Calculate x, y position for a node"""
        if index >= n:
            return None, None
            
        # Find which level this node is on
        level = math.floor(math.log2(index + 1))
        
        # Position within the level
        position_in_level = index - (2 ** level - 1)
        
        # Calculate positions
        y = level * level_height + 10
        
        # X position: center the level, then space nodes evenly
        level_width = base_width
        nodes_in_level = 2 ** level
        spacing = level_width / (nodes_in_level + 1)
        x = spacing * (position_in_level + 1) - node_size // 2
        
        return x, y
    
    # Draw connecting lines first (so they appear behind nodes)
    for i in range(n):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        
        parent_x, parent_y = calculate_position(i)
        if parent_x is None:
            continue
            
        parent_center_x = parent_x + node_size // 2
        parent_center_y = parent_y + node_size // 2
        
        # Draw line to left child
        if left_child < n:
            child_x, child_y = calculate_position(left_child)
            if child_x is not None:
                child_center_x = child_x + node_size // 2
                child_center_y = child_y + node_size // 2
                
                # SVG line for clean connection
                html += f"""
                <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
                    <line x1="{parent_center_x}" y1="{parent_center_y}" 
                          x2="{child_center_x}" y2="{child_center_y}" 
                          stroke="#2c3e50" stroke-width="2"/>
                </svg>
                """
        
        # Draw line to right child
        if right_child < n:
            child_x, child_y = calculate_position(right_child)
            if child_x is not None:
                child_center_x = child_x + node_size // 2
                child_center_y = child_y + node_size // 2
                
                # SVG line for clean connection
                html += f"""
                <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
                    <line x1="{parent_center_x}" y1="{parent_center_y}" 
                          x2="{child_center_x}" y2="{child_center_y}" 
                          stroke="#2c3e50" stroke-width="2"/>
                </svg>
                """
    
    # Draw all nodes
    for i in range(n):
        x, y = calculate_position(i)
        if x is None:
            continue
            
        value = str(arr[i])
        if len(value) > 4:
            value = value[:4]
        
        node_style = get_node_style(i)
        
        html += f"""
        <div style="
            position: absolute;
            left: {x}px;
            top: {y}px;
            width: {node_size}px;
            height: {node_size}px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
            {node_style}
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            z-index: 10;
        ">{value}</div>
        """
    
    html += "</div>"
    return html


def visualize_heap_simple(arr, sorted_indices, current_index, current_color, swap_with=None):
    """Ultra-compact heap visualization as formatted array with tree structure indicators"""
    
    if not arr:
        return "<div>Empty heap</div>"
    
    html = "<div style='font-family: monospace; font-size: 14px; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 0;'>"
    
    # Array representation with level indicators
    html += "<div style='margin-bottom: 10px; font-weight: bold; color: #2c3e50;'>Heap Array:</div>"
    
    level = 0
    level_start = 0
    level_size = 1
    
    while level_start < len(arr):
        level_end = min(level_start + level_size, len(arr))
        
        # Level header
        html += f"<div style='margin: 5px 0; color: #666;'>Level {level}: "
        
        # Elements in this level
        for i in range(level_start, level_end):
            value = arr[i]
            
            # Styling based on state
            if i == current_index:
                style = "background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold;"
            elif i == swap_with:
                style = "background: #f39c12; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold;"
            elif i in sorted_indices:
                style = "background: #27ae60; color: white; padding: 2px 6px; border-radius: 3px;"
            else:
                style = "background: #3498db; color: white; padding: 2px 6px; border-radius: 3px;"
            
            html += f"<span style='{style}'>{value}</span> "
        
        html += "</div>"
        
        level_start = level_end
        level_size *= 2
        level += 1
    
    # Add simple parent-child relationships
    html += "<div style='margin-top: 15px; font-size: 12px; color: #666;'>"
    html += "<div style='font-weight: bold; margin-bottom: 5px;'>Parent-Child Relations:</div>"
    
    for i in range(len(arr)):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        
        if left_child < len(arr) or right_child < len(arr):
            html += f"<div>{arr[i]} → "
            if left_child < len(arr):
                html += f"L:{arr[left_child]} "
            if right_child < len(arr):
                html += f"R:{arr[right_child]}"
            html += "</div>"
    
    html += "</div></div>"
    
    return html
@app.route('/heap_sort/sort', methods=['POST'])
def heap_sort_api():
    data = request.get_json()
    arr = data.get('array', [])
    
    topdown_steps = topDownHeapSort(arr.copy())
    bottomup_steps = bottomUpHeapSort(arr.copy())
    
    # Generate visualizations for each step
    topdown_visualizations = []
    for step in topdown_steps:
        step_arr, sorted_until, current_index, current_color = step
        # Get HTML visualization directly (no need for .pipe())
        html_content = visualize_heap(step_arr, sorted_until, current_index, current_color)
        topdown_visualizations.append({
            'array': step_arr,
            'sorted_indices': list(sorted_until) if isinstance(sorted_until, set) else sorted_until,
            'current_index': current_index,
            'current_color': current_color,
            'visualization': html_content
        })
    
    bottomup_visualizations = []
    for step in bottomup_steps:
        step_arr, sorted_until, current_index, current_color, swap_with = step
        # Get HTML visualization directly (no need for .pipe())
        html_content = visualize_heap(step_arr, sorted_until, current_index, current_color, swap_with)
        bottomup_visualizations.append({
            'array': step_arr,
            'sorted_indices': list(sorted_until) if isinstance(sorted_until, set) else sorted_until,
            'current_index': current_index,
            'current_color': current_color,
            'swap_with': swap_with,
            'visualization': html_content
        })
    
    return jsonify({
        'topdown_steps': topdown_visualizations,
        'bottomup_steps': bottomup_visualizations,
        'sorted_array': sorted(arr),
        'message': 'Heap sort completed with compact tree visualization!'
    })

@app.route('/heap_sort/visualize_step', methods=['POST'])
def visualize_heap_step():
    """Generate visualization for a specific heap sort step"""
    data = request.get_json()
    arr = data.get('array', [])
    sorted_indices = set(data.get('sorted_indices', []))
    current_index = data.get('current_index', -1)
    current_color = data.get('current_color', '')
    swap_with = data.get('swap_with', None)
    
    # Get HTML visualization directly (no need for .pipe())
    html_content = visualize_heap(arr, sorted_indices, current_index, current_color, swap_with)
    
    return jsonify({
        'visualization': html_content,
        'array': arr,
        'sorted_indices': list(sorted_indices),
        'current_index': current_index,
        'current_color': current_color,
        'swap_with': swap_with
    })
@app.route('/heap_sort')
def heap_sort():
    return render_template('heap_sort.html')

@app.route('/graph_coloring')
def graph_coloring():
    return render_template('graph_coloring.html')

@app.route('/graph_coloring/color', methods=['POST'])
def graph_coloring_api():
    data = request.get_json()
    adjacency_matrix = data.get('adjacency_matrix', [])
    
    # Create NetworkX graph from adjacency matrix
    G = nx.Graph()
    n = len(adjacency_matrix)
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)
    
    # Perform coloring
    coloring = greedy_coloring(G)
    is_valid = is_valid_coloring(G, coloring)
    color_count = len(set(coloring.values()))
    
    return jsonify({
        'coloring': coloring,
        'is_valid': is_valid,
        'color_count': color_count,
        'edges': list(G.edges()),
        'nodes': list(G.nodes())
    })

if __name__ == '__main__':
    app.run(debug=True)