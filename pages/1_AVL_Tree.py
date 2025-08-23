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

def visualize_heap(arr, sorted_indices, current_index, current_color, swap_with=None):
    dot = Digraph()
    dot.attr(rankdir='TB')  # Top to bottom layout
    dot.attr('node', shape='circle', style='filled', fontsize='14', fontweight='bold')
    dot.attr('edge', color='#34495e', penwidth='3')  # Dark blue edges with thickness
    
    n = len(arr)
    
    # Add all nodes first
    for i in range(n):
        # Determine node color and style
        if i == current_index:
            node_color = current_color
            node_style = 'filled,bold'
        elif i == swap_with:
            node_color = current_color
            node_style = 'filled,bold'
        elif i in sorted_indices:
            node_color = "#27ae60"  # Green for sorted
            node_style = 'filled'
        else:
            node_color = "#3498db"  # Blue for unsorted
            node_style = 'filled'
        
        label = str(arr[i])
        dot.node(str(i), label, 
                color=node_color, 
                fillcolor=node_color, 
                fontcolor='white',
                style=node_style,
                width='0.8',
                height='0.8')
    
    # Add all parent-child edges to show the complete heap structure
    for i in range(n):
        # Left child: 2*i + 1
        left_child = 2 * i + 1
        if left_child < n:
            dot.edge(str(i), str(left_child))
        
        # Right child: 2*i + 2
        right_child = 2 * i + 2
        if right_child < n:
            dot.edge(str(i), str(right_child))
    
    return dot

def highlight_array(arr, sorted_indices):
    """Helper function to create highlighted array display"""
    highlighted_arr = []
    for i in range(len(arr)):
        if i in sorted_indices:
            highlighted_arr.append(f'<span style="color:green">{arr[i]}</span>')
        else:
            highlighted_arr.append(f'<span style="color:black">{arr[i]}</span>')
    return highlighted_arr

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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/avl_tree')
def avl_tree():
    return render_template('avl_tree.html')

@app.route('/avl_tree/insert', methods=['POST'])
def avl_insert():
    data = request.get_json()
    numbers = data.get('numbers', [])
    
    root = None
    steps = []
    
    for key in numbers:
        root = insert(root, key)
        current_graph = visualize_tree(root, Digraph())
        svg_content = current_graph.pipe(format='svg').decode('utf-8')
        steps.append(svg_content)
    
    ino = []
    pre = []
    inOrder(root, ino)
    preOrder(root, pre)
    
    return jsonify({
        'steps': steps,
        'inorder': ino,
        'preorder': pre
    })

@app.route('/heap_sort')
def heap_sort():
    return render_template('heap_sort.html')

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
        dot = visualize_heap(step_arr, sorted_until, current_index, current_color)
        svg_content = dot.pipe(format='svg').decode('utf-8')
        topdown_visualizations.append({
            'array': step_arr,
            'sorted_indices': list(sorted_until) if isinstance(sorted_until, set) else sorted_until,
            'current_index': current_index,
            'current_color': current_color,
            'visualization': svg_content
        })
    
    bottomup_visualizations = []
    for step in bottomup_steps:
        step_arr, sorted_until, current_index, current_color, swap_with = step
        dot = visualize_heap(step_arr, sorted_until, current_index, current_color, swap_with)
        svg_content = dot.pipe(format='svg').decode('utf-8')
        bottomup_visualizations.append({
            'array': step_arr,
            'sorted_indices': list(sorted_until) if isinstance(sorted_until, set) else sorted_until,
            'current_index': current_index,
            'current_color': current_color,
            'swap_with': swap_with,
            'visualization': svg_content
        })
    
    return jsonify({
        'topdown_steps': topdown_visualizations,
        'bottomup_steps': bottomup_visualizations,
        'sorted_array': sorted(arr),
        'message': 'Heap sort completed with clear parent-child line visualization!'
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
    
    dot = visualize_heap(arr, sorted_indices, current_index, current_color, swap_with)
    svg_content = dot.pipe(format='svg').decode('utf-8')
    
    return jsonify({
        'visualization': svg_content,
        'array': arr,
        'sorted_indices': list(sorted_indices),
        'current_index': current_index,
        'current_color': current_color,
        'swap_with': swap_with
    })

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