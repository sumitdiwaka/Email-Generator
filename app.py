import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import chromadb
import uuid
import json

# Set page configuration and theme
st.set_page_config(
    page_title="Cold Email Generator",
    page_icon="✉️",
    layout="wide",
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #E0E0E0;
    }
    .success-message {
        background-color: #D4EDDA;
        color: #155724;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    load_dotenv()
    return ChatGroq(model="llama-3.3-70b-versatile")

def process_job_url(url, llm):
    loader = WebBaseLoader(url)
    page_data = loader.load().pop().page_content
    
    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing
        following keys: `role`, `experience`, `skills` and `description`.
        Only return valid JSON.
        ### VALID JSON (NO PREAMBLE): 
        """
    )
    
    chain = prompt_extract | llm
    res = chain.invoke(input={'page_data':page_data})
    return JsonOutputParser().parse(res.content)

def generate_email(job_data, portfolio_links, user_info, llm):
    prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### USER INFORMATION:
        Name: {name}
        Job Title: {job_title}
        Company: {company}
        Skills: {skills}
        Experience: {experience}
        Key Interests: {interests}
        
        ### INSTRUCTION:
        You are {name}, a {job_title} at {company}. {company_description}
        Your job is to write a cold email to the client regarding the job mentioned above describing your capabilities
        in fulfilling their needs based on your skills and experience.
        
        Highlight how your skills ({skills}) align with the job requirements.
        Mention your relevant experience: {experience}
        Include your key interests that relate to the position: {interests}
        
        Also add the most relevant ones from the following links to showcase your portfolio: {link_list}
        
        Create a professional, compelling email with:
        1. A catchy subject line
        2. A personalized greeting
        3. A strong introduction
        4. Body that shows alignment between your skills and their needs
        5. Call to action
        6. Professional closing
        
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        """
    )
    
    chain_email = prompt_email | llm
    res = chain_email.invoke({
        "job_description": str(job_data), 
        "link_list": portfolio_links,
        "name": user_info["name"],
        "job_title": user_info["job_title"],
        "company": user_info["company"],
        "company_description": user_info["company_description"],
        "skills": user_info["skills"],
        "experience": user_info["experience"],
        "interests": user_info["interests"]
    })
    return res.content

def main():
    # Initialize session state for reset functionality
    if 'email_generated' not in st.session_state:
        st.session_state.email_generated = False
    if 'generated_email' not in st.session_state:
        st.session_state.generated_email = ""
    if 'reset_requested' not in st.session_state:
        st.session_state.reset_requested = False
    
    # Handle reset logic
    if st.session_state.reset_requested:
        # Clear the reset flag
        st.session_state.reset_requested = False
        st.session_state.email_generated = False
        st.session_state.generated_email = ""
        # No need to modify widget values directly
    
    # Header
    st.markdown('<div class="main-header">✉️ Professional Cold Email Generator</div>', unsafe_allow_html=True)
    st.markdown('Generate personalized cold emails for job applications that highlight your skills and experience')
    
    # Layout with columns for better organization
    col1, col2 = st.columns([1, 2])
    
    # Sidebar for user information
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Your Professional Profile</div>', unsafe_allow_html=True)
        
        # Get default values from session state if they exist
        default_name = "" if st.session_state.reset_requested else st.session_state.get('name', "")
        default_job_title = "" if st.session_state.reset_requested else st.session_state.get('job_title', "")
        default_company = "AtliQ" if st.session_state.reset_requested else st.session_state.get('company', "AtliQ")
        default_company_desc = ("An AI & Software Consulting company dedicated to facilitating the seamless integration of business processes through automated tools. "
                              "Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, "
                              "process optimization, cost reduction, and heightened overall efficiency.") if st.session_state.reset_requested else st.session_state.get('company_description', 
                              "An AI & Software Consulting company dedicated to facilitating the seamless integration of business processes through automated tools. "
                              "Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, "
                              "process optimization, cost reduction, and heightened overall efficiency.")
        default_skills = "" if st.session_state.reset_requested else st.session_state.get('skills', "")
        default_experience = "" if st.session_state.reset_requested else st.session_state.get('experience', "")
        default_interests = "" if st.session_state.reset_requested else st.session_state.get('interests', "")
        
        user_name = st.text_input("Your Full Name", value=default_name, key="name")
        user_job_title = st.text_input("Your Job Title", value=default_job_title, key="job_title")
        user_company = st.text_input("Your Company", value=default_company, key="company")
        user_company_description = st.text_area(
            "Company Description", 
            value=default_company_desc,
            key="company_description"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        user_skills = st.text_area("Your Key Skills", value=default_skills, key="skills", 
                                  help="List your technical and soft skills relevant to the position")
        user_experience = st.text_area("Your Relevant Experience", value=default_experience, key="experience",
                                      help="Highlight your past work experience, projects, or achievements")
        user_interests = st.text_area("Your Professional Interests", value=default_interests, key="interests",
                                     help="Areas of interest that align with the job position")
        
        # Reset button in sidebar
        if st.button("Reset All Fields", type="secondary"):
            st.session_state.reset_requested = True
            st.rerun()
    
    # Left column - Job Information Input
    with col1:
        st.markdown('<div class="sub-header">Job Information</div>', unsafe_allow_html=True)
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📋 Job URL", "✏️ Manual Entry"])
        
        # Job URL tab
        with tab1:
            default_job_url = "" if st.session_state.reset_requested else st.session_state.get('job_url', "")
            job_url = st.text_input("Enter Job Posting URL:", value=default_job_url, key="job_url",
                                   help="Paste the complete URL of the job posting")
            submit_url = st.button("Generate Email from URL", type="primary")
        
        # Manual entry tab
        with tab2:
            default_manual_role = "" if st.session_state.reset_requested else st.session_state.get('manual_role', "")
            default_manual_exp = "" if st.session_state.reset_requested else st.session_state.get('manual_experience', "")
            default_manual_skills = "" if st.session_state.reset_requested else st.session_state.get('manual_skills', "")
            default_manual_desc = "" if st.session_state.reset_requested else st.session_state.get('manual_description', "")
            
            manual_role = st.text_input("Job Role:", value=default_manual_role, key="manual_role")
            manual_experience = st.text_input("Required Experience:", value=default_manual_exp, key="manual_experience")
            manual_skills = st.text_area("Required Skills:", value=default_manual_skills, key="manual_skills")
            manual_description = st.text_area("Job Description:", value=default_manual_desc, key="manual_description")
            submit_manual = st.button("Generate Email from Manual Entry", type="primary")
    
    # Right column - Results
    with col2:
        if st.session_state.email_generated:
            st.markdown('<div class="sub-header">Your Generated Email</div>', unsafe_allow_html=True)
            st.markdown('<div class="success-message">✅ Email successfully generated! You can edit it below:</div>', 
                      unsafe_allow_html=True)
            
            edited_email = st.text_area("Email Content", st.session_state.generated_email, 
                                      height=400, key="edited_email")
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                st.download_button(
                    label="Download Email",
                    data=edited_email,
                    file_name="cold_email.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_btn2:
                if st.button("Copy to Clipboard", use_container_width=True):
                    st.code(edited_email)
                    st.success("Email copied! Use the code block above to copy the email.")
            with col_btn3:
                if st.button("Generate New Email", type="secondary", use_container_width=True):
                    st.session_state.email_generated = False
                    st.rerun()
    
    # Process the job information and generate email
    if submit_url or submit_manual:
        # Validate user inputs
        if not user_name or not user_job_title:
            st.error("Please provide your name and job title in the sidebar.")
            return
            
        # Collect user information
        user_info = {
            "name": user_name,
            "job_title": user_job_title,
            "company": user_company,
            "company_description": user_company_description,
            "skills": user_skills,
            "experience": user_experience,
            "interests": user_interests
        }
        
        with st.spinner("Processing... This may take a moment"):
            try:
                # Initialize models and DB
                llm = load_models()
                client = chromadb.PersistentClient('vector_db_store')
                collection = client.get_or_create_collection(name="portfolio")
                
                # Load portfolio if DB is empty
                if not collection.count():
                    df = pd.read_csv("my_portfolio.csv")
                    for _, row in df.iterrows():
                        collection.add(
                            documents=row['Techstack'],
                            metadatas={"links": row["Links"]},
                            ids=[str(uuid.uuid4())]
                        )
                
                # Process job information
                if submit_url:
                    job_data = process_job_url(job_url, llm)
                    # Display extracted job information
                    with col1:
                        st.success("Successfully extracted job details!")
                        with st.expander("View Extracted Job Details"):
                            st.json(job_data)
                else:
                    # Create job data from manual entry
                    job_data = {
                        "role": manual_role,
                        "experience": manual_experience,
                        "skills": manual_skills,
                        "description": manual_description
                    }
                
                # Get relevant portfolio links based on skills
                query_text = job_data.get('skills', '')
                if query_text:
                    links = collection.query(query_texts=query_text, n_results=2).get('metadatas', [])
                else:
                    links = []
                
                # Generate email
                email = generate_email(job_data, links, user_info, llm)
                st.session_state.generated_email = email
                st.session_state.email_generated = True
                
                # Force a rerun to show the results section
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.email_generated = False

if __name__ == "__main__":
    main()