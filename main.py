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
    """, unsafe_allow_html=True
)

# Create columns for the buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("AVL Tree Visualizer", key="avl"):
        st.switch_page("pages/1_AVL_Tree.py")

with col2:
    if st.button("Heap Sort Visualizer", key="heap"):
        st.switch_page("pages/2_Heap_Sort.py")

with col3:
    if st.button("Graph Coloring Visualizer", key="graph"):
        st.switch_page("pages/3_Graph_Coloring.py")
