import streamlit as st
from main import get_pdf_text, get_text_chunks, generate_lesson_plan, create_vector_store
from langchain.schema import Document

# Streamlit application
st.set_page_config(page_title="Lesson Plan Generator", page_icon=":books:", layout="wide")

st.title("Lesson Planner")

# Create tabs
tabs = st.tabs(["Instructions", "Planners"])

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload Your Document", type=["pdf"], label_visibility="visible")
st.sidebar.markdown("### Connect with me on [LinkedIn](https://www.linkedin.com/in/sreemurali-sekar-k-84630517a/)")

# Tab 1: Instructions
with tabs[0]:
    st.header("Instructions")
    st.write("1. Upload a PDF file containing the lesson content in the sidebar.")
    st.write("2. Enter a descriptive lesson title.")
    st.write("3. Select the appropriate grade level.")
    st.write("4. Click the 'Generate Lesson Plan' button to create your lesson plan.")

# Tab 2: Planner
with tabs[1]:
    
    # Text input for lesson title with placeholder and help text
    lesson_title = st.text_input("Lesson Title", placeholder="Enter the title of the lesson", help="Provide a descriptive title for the lesson.")
    
    # Select box for grade level with a default option
    grade_level = st.selectbox("Grade Level", ["Select Grade", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade", "10th Grade", "11th Grade", "12th Grade"])
    
    # Generate button with a confirmation message
    if st.button("Generate Lesson Plan"):
        if uploaded_file and lesson_title and grade_level and grade_level != "Select Grade":
            with st.spinner("Generating lesson plan..."):
                # Read the file content
                file_content = uploaded_file.read()
                
                # Extract text from PDF
                raw_text = get_pdf_text(file_content)
                
                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vector_store = create_vector_store(text_chunks)
                
                # Retrieve the context as documents
                context = [Document(page_content=chunk) for chunk in text_chunks]
                user_query = f'Generate a lesson planner for the lesson title {lesson_title} with following grade level {grade_level}'
                
                # Generate the lesson plan
                response = generate_lesson_plan(context, user_query)
                
                st.write(response)
        else:
            st.warning("Please ensure all fields are filled in correctly: enter a lesson title, select a grade level, and upload a PDF file in the sidebar.")
