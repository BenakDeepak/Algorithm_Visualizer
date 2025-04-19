import streamlit as st 
import networkx as nx 
import matplotlib.pyplot as plt 
import random

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

def plot_graph(graph, coloring): 
    pos = nx.spring_layout(graph, k=0.3) 
    color_map = [coloring.get(node, 0) for node in graph.nodes()] 
    plt.figure(figsize=(8, 6)) 
    nx.draw(graph, pos, node_color=color_map, node_size=800, font_color='black', with_labels=True, cmap=plt.cm.coolwarm) 
    plt.title("Graph Coloring") 
    st.pyplot(plt)
    plt.clf()  # Clear the plot after displaying it to avoid reuse issues

def generate_random_graph(n_nodes, edge_prob): 
    G = nx.Graph() 
    G.add_nodes_from(range(n_nodes)) 
    for i in range(n_nodes): 
        for j in range(i + 1, n_nodes): 
            if random.random() < edge_prob: 
                G.add_edge(i, j) 
    return G

def main(): 
    st.title("Graph Coloring Visualization") 
    option = st.selectbox("Choose graph input method", ["Manual Input", "Random Graph"]) 
    
    if option == "Manual Input": 
        n_nodes = st.number_input("Enter the number of nodes", min_value=1, step=1, key="nodes") 

        G = nx.Graph() 
        G.add_nodes_from(range(n_nodes)) 

        matrix_text = st.text_area("Enter the adjacency matrix (row by row, separated by new lines):") 
        if matrix_text: 
            try: 
                adj_matrix = [list(map(int, row.split())) for row in matrix_text.strip().split('\n')] 
                
                if len(adj_matrix) != n_nodes or any(len(row) != n_nodes for row in adj_matrix): 
                    st.error("Adjacency matrix dimensions do not match the number of nodes.") 
                else: 
                    for i in range(n_nodes):  # Corrected 'n_node' to 'n_nodes'
                        for j in range(i + 1, n_nodes): 
                            if adj_matrix[i][j] == 1: 
                                G.add_edge(i, j) 
                    if st.button("Perform Greedy Coloring"): 
                        coloring_result = greedy_coloring(G) 
                        st.write('Coloring: ', coloring_result) 
                        st.write('Valid: ', is_valid_coloring(G, coloring_result)) 
                        st.write('Number of Colors Used (K): ', len(set(coloring_result.values()))) 
                        plot_graph(G, coloring_result) 
            except ValueError: 
                st.error("Invalid input format. Please enter integers separated by spaces.") 
    elif option == "Random Graph": 
        n_nodes = st.number_input("Enter the number of nodes", min_value=1, step=1, key="rand_nodes") 
        edge_prob = st.slider("Enter the edge probability (0.0 to 1.0)", 0.0, 1.0, 0.5) 
        if st.button("Perform Greedy Coloring"): 
            G = generate_random_graph(n_nodes, edge_prob) 
            st.write(f"Generated a random graph with {n_nodes} nodes and edge probability {edge_prob:.2f}") 
            coloring_result = greedy_coloring(G) 
            st.write('Coloring: ', coloring_result) 
            st.write('Valid: ', is_valid_coloring(G, coloring_result)) 
            st.write('Number of Colors Used (K): ', len(set(coloring_result.values()))) 
            plot_graph(G, coloring_result) 

if __name__ == "__main__": 
    main()