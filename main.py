import streamlit as st

# Title of the application
st.markdown(
    """
    <h1 style="text-align: center;">Algorithm Visualizers</h1>
    """, unsafe_allow_html=True
)

# Create a container for the buttons using Markdown with flexbox
st.markdown(
    """
    <style>
        .button {
            padding: 15px;  /* Increased padding for larger buttons */
            background-color: grey;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 18px;  /* Increased font size */
            width: 250px;  /* Increased button width */
            border-radius: 8px;  /* More rounded corners */
            transition: background-color 0.3s, transform 0.3s;  /* Smooth transition */
            margin: 20px;  /* Margin between buttons */
            margin-left: 0;  /* Remove left margin from first button */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Added shadow effect */
            text-align: center;  /* Center text */
        }
        .button:hover {
            background-color: #555;  /* Darker shade on hover */
            transform: translateY(-4px);  /* Slight lift effect */
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);  /* Shadow deepens on hover */
        }
    </style>
    
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <a href="https://avl-tree.onrender.com" target="_blank">
            <button class="button">
                AVL Tree Visualizer
            </button>
        </a>
        <a href="https://heap-sort-visualizers.onrender.com" target="_blank">
            <button class="button">
                Heap Sort Visualizer
            </button>
        </a>
        <a href="https://graph-coloring.onrender.com" target="_blank">
            <button class="button">
                Graph Coloring Visualizer
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True
)
