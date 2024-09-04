import streamlit as st
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from markdown import markdown
import tempfile
from pydub import AudioSegment

# Set your OpenAI API key here
openai.api_key = "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"

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # This is the dimension of the embeddings from the all-MiniLM-L6-v2 model
index = faiss.IndexFlatL2(dimension)

# Dictionary to store company profiles
company_profiles = {}

def load_md_file(file):
    content = file.getvalue().decode("utf-8")
    return content

def create_embedding(text):
    return model.encode([text])[0]

def add_to_index(embedding, idx):
    index.add(np.array([embedding]))
    return idx + 1

def search_similar_profiles(query, k=1):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)
    return I[0]

def generate_email(company_profile, supplier_info):
    prompt = f"""
    You are an experienced sales and marketing professional specializing in the Ready Made Garments (RMG) industry. Your task is to craft a personalized email to a potential buyer based on their company profile. Use the following structure to create a compelling and tailored email:

    Company Profile:
    {company_profile}

    Supplier Information:
    {supplier_info}

    Based on the company profile and supplier information provided above, craft a personalized email following the structure and guidelines below:


    Subject Line: Create an attention-grabbing subject line that addresses a key pain point or requirement.
    Greeting: Address the sourcing manager by name.
    Introduction: Briefly introduce yourself and your company.
    Acknowledgment: Show understanding of the buyer's business and their relationship with Bangladesh.
    Value Proposition: Explain how your company can address their specific pain points and meet their requirements.
    Unique Selling Points: Highlight 2-3 key ways your company stands out in addressing their needs.
    Call to Action: Suggest a next step, such as a phone call or meeting.
    Closing: Professional and courteous sign-off.


    Guidelines:

    Keep the email concise and to the point (aim for 200-250 words).
    Use a professional yet conversational tone.
    Demonstrate knowledge of their business without being presumptuous.
    Focus on how you can solve their problems rather than just listing your company's achievements.
    Personalize the email with specific details from their profile.
    Avoid generic statements that could apply to any company.


    Example Output:
    Subject: [Attention-grabbing subject line]
    Dear [Sourcing Manager's Name],
    [Introduction and acknowledgment of their business]
    [Value proposition addressing specific pain points]
    [Unique selling points of your company]
    [Call to action]
    [Professional closing]
    [Your name]
    [Your position]
    [Your company name
    """

    client = openai.OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that writes personalized business emails."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

def chatbot_response(user_input):
    # Search for relevant company profiles
    similar_indices = search_similar_profiles(user_input, k=3)
    context = "\n\n".join([company_profiles[i] for i in similar_indices if i in company_profiles])

    prompt = f"""
    You are an AI assistant specializing in garment buyer companies. Use the provided company profile information {context} to answer the user's question: {user_input}. Respond with a single, specific sentence that directly addresses the question. If the exact information isn't available, provide the most relevant fact from the context. If no relevant information exists, simply state "That information is not available in the company profile." Do not infer or create information not present in the context.
    """

    client = openai.OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant knowledgeable about garment buyer companies."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,
        n=1,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()
def summarize_text(text):
    prompt = f"Please summarize the following text in 2-3 sentences:\n\n{text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        temperature=0.7,
    )

    return response.choices[0].message["content"].strip()

#Audio Transcription

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
    return response["text"]

def main():
    st.title("Buyer Approach Application with Chatbot")

    # File uploader for company profiles
    uploaded_files = st.file_uploader("Upload company profile .md files", accept_multiple_files=True, type=['md'])

    idx = 0
    for file in uploaded_files:
        content = load_md_file(file)
        embedding = create_embedding(content)
        idx = add_to_index(embedding, idx)
        company_profiles[idx-1] = content
        st.success(f"Processed: {file.name}")

    # Tabs for different functionalities
    tab1, tab2,tab3 = st.tabs(["Email Generator", "Company Chatbot","Audio Transcription"])

    with tab1:
        st.header("Email Generator")
        st.subheader("Your Company Information")

        # Add option to choose between manual input and file upload
        input_method = st.radio("Choose input method:", ["Manual Input", "Upload .md File"])

        if input_method == "Manual Input":
            supplier_name = st.text_input("Company Name")
            supplier_location = st.text_input("Location")
            supplier_specialization = st.text_input("Specialization")
            supplier_experience = st.text_input("Years of Experience")

            supplier_info = f"""
            Company Name: {supplier_name}
            Location: {supplier_location}
            Specialization: {supplier_specialization}
            Years of Experience: {supplier_experience}
            """
        else:
            uploaded_supplier_file = st.file_uploader("Upload your company profile .md file", type=['md'])
            if uploaded_supplier_file is not None:
                supplier_info = load_md_file(uploaded_supplier_file)
                st.success(f"Processed: {uploaded_supplier_file.name}")
                st.markdown("### Your Company Profile")
                st.markdown(supplier_info)
            else:
                supplier_info = ""

        # Input for buyer search
        st.subheader("Search for a Buyer")
        
        # Create a list of company names
        company_names = [content.split('\n')[0].strip() for content in company_profiles.values()]
        
        # Add dropdown for company selection
        selected_company = st.selectbox("Select a company", [""] + company_names)
        
        if st.button("Generate Email"):
            if selected_company and supplier_info:
                # Find the matching profile for the selected company
                matching_profile = next((profile for profile in company_profiles.values() if profile.startswith(selected_company)), None)
                
                if matching_profile:
                    st.subheader("Selected Company Profile")
                    st.markdown(matching_profile)

                    email_content = generate_email(matching_profile, supplier_info)
                    st.subheader("Generated Email")
                    st.text_area("Email Content", email_content, height=300)
                else:
                    st.error("Could not find the selected company profile. Please try again.")
            else:
                st.warning("Please select a company and provide your company information.")

    with tab2:
        st.header("Company Chatbot")
        st.write("Ask me anything about the companies in the database!")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What would you like to know?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = chatbot_response(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
       
            # Generate and display summary of the response
            summary = summarize_text(response)
            st.subheader("Response Summary")
            st.write(summary)
       
        with tab3:
            st.header("Audio Transcriber")
            uploaded_audio = st.file_uploader("Upload an audio file for transcription", type=['mp3', 'wav', 'm4a'])

            if uploaded_audio is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_file.write(uploaded_audio.read())
                    temp_file_path = temp_file.name
                
                audio_format = uploaded_audio.name.split('.')[-1]
                if audio_format not in ['wav', 'mp3']:
                    audio = AudioSegment.from_file(temp_file_path, format=audio_format)
                    temp_file_path = temp_file_path.replace(audio_format, 'wav')
                    audio.export(temp_file_path, format='wav')
                
                transcription = transcribe_audio(temp_file_path)
                st.subheader("Transcription")
                st.text_area("Transcribed Text", transcription, height=300)
            
            

if __name__ == "__main__":
    main()
