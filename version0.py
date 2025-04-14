import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from autogen import AssistantAgent, UserProxyAgent
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma  

load_dotenv()
os.environ["AUTOGEN_USE_DOCKER"] = "False"

chroma_client = chromadb.PersistentClient(path="chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="documents", embedding_function=sentence_transformer_ef)

class ChatAgent(UserProxyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.responses = []

    def receive(self, message, sender, request_reply=None, silent=False):
        super().receive(message, sender, request_reply, silent=True)
        if isinstance(message, dict) and "content" in message:
            self.responses.append(message["content"])

if "messages" not in st.session_state:
    st.session_state.update({
        "messages": [],
        "custom_keywords": [],
        "current_analysis": "",
        "document_id": None,
        "audience": None,
        "awaiting_custom_keywords": False,
        "custom_keywords_entered": False,
        "article_generated": False,
        "awaiting_refinement": False,
        "refinement_requested": False,
        "post_type": None,
        "tone": None,
        "custom_tone": None,
        "awaiting_custom_tone": False,
        "versions_generated": False,
        "current_article_content": None,
        "uploaded_file": None,
        "file_processed": False,
        "analysis_title": None,
        "analysis_keywords": [],
        "ai_generated_post": None,
        "humanized_post": None
    })

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def process_uploaded_file():
    if not st.session_state.uploaded_file:
        return False
    
    try:
        text = extract_text_from_pdf(st.session_state.uploaded_file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        document_id = st.session_state.uploaded_file.name
        collection.add(
            documents=chunks[:3],
            ids=[f"{document_id}_chunk{i}" for i in range(min(3, len(chunks)))]
        )
        
        st.session_state.document_id = document_id
        st.session_state.file_processed = True
        return True
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False

def extract_title_keywords_relevance(analysis_text):
    lines = analysis_text.split("\n")
    title = ""
    keywords = []
    scores = {}
    in_keywords = False
    in_scores = False
    for line in lines:
        line = line.strip()
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("keywords:"):
            in_keywords = True
            continue
        elif line.lower().startswith("relevance analysis:") or "rating" in line.lower():
            in_keywords = False
            in_scores = True
            continue
        if in_keywords and "*" in line:
            keywords.extend([kw.strip("* ").strip() for kw in line.split("*") if kw.strip()])
        if in_scores and (line.startswith("-") or line.startswith("*")):
            try:
                clean_line = line.lstrip("-* ").strip()
                if ":" in clean_line:
                    kw_part, rest = clean_line.split(":", 1)
                    if "%" in rest:
                        score = int(rest.split("%")[0].strip().split()[-1])
                        scores[kw_part.strip()] = score
            except Exception:
                continue
    filtered_keywords = [kw for kw in keywords if scores.get(kw, 0) >= 50]
    return title, filtered_keywords

def generate_analysis():
    if not st.session_state.file_processed:
        st.error("Please upload and process a file first")
        return False
    try:
        results = collection.get(ids=[f"{st.session_state.document_id}_chunk{i}" for i in range(3)])
        content = "\n\n".join(results["documents"])
        assistant = AssistantAgent(
            name="seo_analyst",
            system_message="""Generate content in this EXACT format:

                Title: [Your engaging title here]

                Keywords:
                * [keyword1] * [keyword2] * [keyword3] 
                * [keyword4] * [keyword5] * [keyword6]

                Relevance Analysis:
                - keyword1: explanation. -  100%
                - keyword2: explanation. -  85%
                - keyword3: explanation. -  45%
                - if the keywords is not relevant means say them as zero.
                [and so on for all keywords]""",
            llm_config={
                "config_list": [{
                    "model": "gemini-1.5-flash",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "api_type": "google"
                }],
                "temperature": 0.7
            }
        )
        user_proxy = ChatAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1)
        num_additional = 15 - len(st.session_state.custom_keywords)
        task = f"""
            For {st.session_state.audience} audience, provide:
            1. First provide 1 engaging title (12 words max)
            2. Then list all 15 keywords (including custom and additional) in this exact format:
            * [keyword1] * [keyword2] * ...
            3. For each keyword:
            - Write a short explanation
            - Rate relevance %
            - Format each like: - keyword: explanation. – XX%
            - Sort all keywords by rating descending

            Custom keywords to include: {', '.join(st.session_state.custom_keywords)}
            Add {num_additional} more relevant keywords (must be different)

            Document Content:
            {content}
        """
        user_proxy.initiate_chat(assistant, message=task, clear_history=True)

        if user_proxy.responses:
            response = user_proxy.responses[-1]
            cleaned_response = response.replace("```", "")  
            cleaned_response = "\n".join(line.lstrip() for line in cleaned_response.splitlines()) 
            st.session_state.current_analysis = cleaned_response
            title, keywords = extract_title_keywords_relevance(cleaned_response)
            st.session_state.analysis_title = title
            st.session_state.analysis_keywords = keywords
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**Analysis with Your Keywords:**\n\n{cleaned_response}"
            })
            return True
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return False

def generate_article():
    try:
        title, keywords = extract_title_keywords_relevance(st.session_state.current_analysis)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="chroma_db"
        )
        relevant_docs = vector_store.similarity_search(f"{title}. Keywords: {', '.join(keywords)}", k=3)
        relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
            Write a clear, engaging article (500-600 words) on: {title}
            Make it simple, crisp, and easy to follow for a broad audience.

            **STRUCTURE & STYLE**
            1. **Introduction** – Start with a bold claim, surprising stat, or thought-provoking question.
            2. **Body** – Use short paragraphs and clear subheadings. Include:
                - Big Picture (why it matters)
                - Practical Impacts (real-world relevance)
                - Simplified Technical Insights
            3. **Content Quality**
                - Use analogies and real examples
                - Include 2-3 relevant facts or stats
                - Naturally weave in keywords: {', '.join(keywords)}
            4. **Tone**
                - Professional yet conversational
                - No jargon, no fluff
            5. **Conclusion**
                - Summarize key points
                - Share future implications or prompt reflection

            Reference this content:
            {relevant_content}
            """
        assistant = AssistantAgent(
            name="writer_agent",
            system_message="You're a professional content writer.",
            llm_config={
                "config_list": [{
                    "model": "gemini-1.5-flash",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "api_type": "google"
                }],
                "temperature": 0.6
            }
        )
        user = ChatAgent(name="user", human_input_mode="NEVER", max_consecutive_auto_reply=1)
        user.initiate_chat(assistant, message=prompt, clear_history=True)
        if user.responses:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"###  Your Article \n\n{user.responses[-1]}"
            })
            st.session_state.current_article_content = user.responses[-1]
            st.session_state.article_generated = True
            return True
    except Exception as e:
        st.error(f"Article generation failed: {str(e)}")
        return False

def refine_article(feedback):
    try:
        current_article = next(
            (msg["content"].split("### Your Article \n\n")[-1].split("### Refined Article\n\n")[0].strip()
             for msg in reversed(st.session_state.messages)
             if msg["role"] == "assistant" and (" Your Article" in msg["content"] or "Refined Article" in msg["content"])),
            None
        )
        if not current_article:
            st.error("Article not found")
            return False
        assistant = AssistantAgent(
            name="refinement_agent",
            system_message="""You're an expert editor that improves articles while maintaining:
            1. Original structure and key information
            2. Professional yet engaging tone
            3. All requested changes from user feedback""",
            llm_config={
                "config_list": [{
                    "model": "gemini-1.5-flash",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "api_type": "google"
                }],
                "temperature": 0.5
            }
        )
        prompt = f"""
        REFINEMENT REQUEST:
        {feedback}

        CURRENT ARTICLE:
        {current_article}

       GUIDELINES FOR REFINEMENT:
        1. Make only the requested changes - don't modify other parts
        2. Keep the same overall structure and tone
        3. Maintain all key facts and information
        4. Highlight changes by bolding new or modified text
        
        OUTPUT REQUIREMENTS:
        - Return the complete revised article
        - Mark changes in bold
        - Keep the same approximate length
        - Maintain all original section headings
        """

        user = ChatAgent(name="user", human_input_mode="NEVER", max_consecutive_auto_reply=1)
        user.initiate_chat(assistant, message=prompt, clear_history=True)
        if user.responses:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"### 🔄 Refined Article\n\n{user.responses[-1]}\n\n*Changes made based on your feedback*"
            })
            return True
    except Exception as e:
        st.error(f"Refinement failed: {str(e)}")
        return False

def generate_post_with_human_version(post_type):
    """Generate platform-optimized content using session state settings"""
    try:
        content = st.session_state.get("current_article_content")
        if not content:
            st.error("No article content found")
            return None
        audience = st.session_state.get("audience", "professionals")
        base_tone = st.session_state.get("tone")
        custom_tone = st.session_state.get("custom_tone")
        if not all([audience, base_tone]):
            st.error("Missing audience or tone selection")
            return None
        selected_tone = custom_tone.lower() if base_tone == "Custom" else base_tone.lower()
        temperature = 0.7 if selected_tone in ["conversational", "friendly"] else 0.5
        post_configs = {
            "Twitter Thread": {
                "max_chars": 280,
                "description": "EXACTLY 1 tweet (280 characters MAX)",
                "prompt": f"""
                Write ONE engaging tweet based on this content for {audience}:
                {content}
                
                ### Requirements:
                - Tone: {selected_tone}
                - STRICT CHARACTER LIMIT: 280 (including spaces)
                - **Front-load key message** in the first 3-4 words.
                - Spark **emotion, passion, or excitement**.
                - Add a **clear CTA** (reply, click, share).
                - Use **2-5 hashtags for reach.
                - Tag someone or add a **link** if relevant.
                - Ensure it's **engaging & optimized for interaction**.
                - **Content has to be like human written and more crisp short etc**
                
                ### Tone Guidelines:
                {"- Casual, friendly tone with emojis" if "casual" in selected_tone else ""}
                {"- Professional but concise" if "formal" in selected_tone else ""}
                {"- Fresh, modern language" if "modern" in selected_tone else ""}
                {f"- Custom tone: {custom_tone}" if base_tone == "Custom" else ""}
                
                Return ONLY the tweet content.
                """
            },
            "Blog Post": {
                "min_words": 300,
                "max_words": 400,
                "description": "300-400 word blog post",
                "prompt": f"""
                Write a blog post based on this content for {audience}:
                {content}
                
                ### Requirements:
                - Tone: {selected_tone}
                - Length: 300-400 words
                - **Hook the Reader** – Start with a bold statement or surprising fact
                - **Engaging Structure** – Use subheadings, bullet points, and short paragraphs(5 lines)
                - **Fresh Insights** – Focus on unique perspectives and real-world impact
                - **Conversational Style** – Keep it {selected_tone} and jargon-free
                - **Credibility** – Back insights with data or examples
                - **Call to Action** – End with a discussion prompt
                - **Content has to be like human written and more crisp short etc**
                - **Use SEO Optimization**
                
                ### Tone Enhancements:
                {"- Conversational with occasional emojis" if "casual" in selected_tone else ""}
                {"- Academic references where appropriate" if "formal" in selected_tone else ""}
                {"- Current terminology and fresh analogies" if "modern" in selected_tone else ""}
                {f"- Follow custom tone: {custom_tone}" if base_tone == "Custom" else ""}
                
                Return ONLY the blog post content.
                """
            },
            "LinkedIn Post": {
                "max_chars": 300,
                "description": "250-300 word LinkedIn post",
                "prompt": f"""
                Write a LinkedIn post based on this content for {audience}:
                {content}
                
                ### Requirements:
                - Tone: {selected_tone}
                - **Hook:** Start with a bold statement, surprising fact, or a relatable question (1 line).
                - **Make It Skimmable:** Use short sentences, line breaks, and bold key points for easy reading.
                - **Explain Simply:** Describe the concept in a crisp, easy-to-understand way (1 line).
                - **Add a Quick Analogy or Example:** Make it relatable with a simple comparison (1 line).
                - **Call to Action:** End with a thought-provoking question to spark discussion (1 line).
                - **Use Hashtags:** Add relevant hashtags at the end.
                - **Content has to be like human written and more crisp short etc**
                - Length: 250-300 words
               
                
                ### Tone Adjustments:
                {"- Slightly more casual with emojis" if "casual" in selected_tone else ""}
                {"- Industry-specific terminology" if "formal" in selected_tone else ""}
                {"- Trend-aware language" if "modern" in selected_tone else ""}
                {f"- Custom tone: {custom_tone}" if base_tone == "Custom" else ""}
                
                Return ONLY the LinkedIn post content.
                """
            },
            "Email Newsletter": {
                "min_words": 300,
                "max_words": 320,
                "description": "300-320 word email",
                "prompt": f"""
                Write an email based on this content for {audience}:
                {content}
                
                ### Requirements:
                - Tone: {selected_tone}
                - Length: 300-320 words
                - **Make It More Personalized** (e.g., "Dear [Name],")
                - Write an engaging, **scannable email** with:
                  - **Compelling hook** (question or bold statement)
                  - **Short paragraphs & bullet points** for readability
                  - **Inverted pyramid structure** leading to a strong **CTA**
                - Ensure a **smooth, conversational flow**.
                - Write a **clear, action-driven CTA** that encourages interaction.
                - Optimize for **mobile readability** & **avoid spam triggers**.
                - Keep it **engaging, relevant & thought-provoking**.
                - **Content has to be like human written and more crisp short etc**
                
                
                ### Tone Customization:
                {"- Warm and approachable" if "casual" in selected_tone else ""}
                {"- Polished and professional" if "formal" in selected_tone else ""}
                {"- Contemporary phrasing" if "modern" in selected_tone else ""}
                {f"- Custom tone: {custom_tone}" if base_tone == "Custom" else ""}
                
                Format:
                Subject: [subject line]
                
                [email body]
                
                Return ONLY the email content.
                """
            }
        }
        config = post_configs.get(post_type)  
        if not config:
            st.error(f"Unsupported post type: {post_type}")
            return None
        ai_agent = AssistantAgent(
            name="platform_specialist",
            system_message=f"""You create {post_type} posts that MUST:
            1. Follow {config['description']}
            2. Maintain {selected_tone} tone
            3. Use platform-appropriate formatting
            4. Never exceed length limits""",
            llm_config={
                "config_list": [{
                    "model": "gemini-1.5-flash",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "api_type": "google"
                }],
                "temperature": temperature
            }
        )
        user_proxy = ChatAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1)
        user_proxy.initiate_chat(ai_agent, message=config["prompt"], clear_history=True)
        ai_post = user_proxy.responses[-1] if user_proxy.responses else None
        if not ai_post:
            st.error("AI post generation failed.")
            return None
        humanizer = AssistantAgent(
            name="humanizer",
            system_message=f"""You transform AI content into natural, human-sounding text while:
            1. Maintaining the {selected_tone} tone
            2. Preserving all key information
            3. Improving flow and readability
            4. Adding natural phrasing
            5. Keeping original format and length""",
            llm_config={
                "config_list": [{
                    "model": "gemini-1.5-flash",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "api_type": "google"
                }],
                "temperature": min(temperature + 0.1, 0.9) 
            }
        )
        humanize_prompt = f"""
        Improve this {post_type} post to sound more human:
        - Keep same length and format
        - Maintain {selected_tone} tone
        -  Rewrite the content below so it sounds natural, human, and engaging—like something you'd say to a friend. 
            Keep the original message, but make it flow effortlessly with personality and a conversational tone.
            Guidelines:
            - Mix short, punchy lines with longer ones.
            - Use smooth, natural transitions.
            - Add light personality or perspective where it fits.
            - Avoid sounding robotic, repetitive, or overly formal.
            - No fluff. No filler. Just clean, real-sounding writing.
            Don’t:
            - Mention it's rewritten.
            - Add or change facts.
            Return only the rewritten version—no extra notes.
        Original content:
        {ai_post}
        Return ONLY the improved version.
        """
        user_proxy.initiate_chat(humanizer, message=humanize_prompt, clear_history=True)
        humanized_post = user_proxy.responses[-1] if user_proxy.responses else None
        return {
            "ai": ai_post,
            "human": humanized_post,
            "config": config
        }
    except Exception as e:
        st.error(f"Post generation failed: {str(e)}")
        return None

st.set_page_config(page_title="Interactive Chat", layout="centered")
st.title("💬 Content Generator")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.document_id:
    with st.chat_message("assistant"):
        st.markdown("Please upload your PDF document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        with st.spinner("Processing document..."):
            if process_uploaded_file():
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Document uploaded! Who is your target audience?"
                })
                st.rerun()

elif not st.session_state.audience:
    with st.chat_message("assistant"):
        st.markdown("Who is your target audience?")
    audience = st.radio(
        "Select audience:",
        ["General Public", "Business Leaders", "Policy Makers", 
         "Investors", "Media", "Sales Teams", "Marketing Professionals"],
        index=None,
        label_visibility="collapsed"
    )
    if audience:
        st.session_state.audience = audience
        st.session_state.messages.append({
            "role": "user", 
            "content": f"Audience: {audience}"
        })
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "What custom keywords would you like to include? (Enter below and press Enter)"
        })
        st.session_state.awaiting_custom_keywords = True
        st.rerun()

elif st.session_state.awaiting_custom_keywords and not st.session_state.custom_keywords_entered:
    with st.chat_message("assistant"):
        st.markdown("Enter up to 5 keywords separated by commas:")
    keywords_input = st.text_input(
        "Keywords", 
        label_visibility="collapsed",
        placeholder="e.g., marketing, strategy, digital, growth, analytics"
    )
    
    if keywords_input:
        st.session_state.messages.append({"role": "user", "content": keywords_input})
        custom_keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
        unique_keywords = []
        for kw in custom_keywords:
            if kw.lower() not in [k.lower() for k in unique_keywords] and len(unique_keywords) < 5:
                unique_keywords.append(kw)
        st.session_state.custom_keywords = unique_keywords
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Thanks! I'll use these keywords: **{', '.join(unique_keywords)}**"
        })
        st.session_state.custom_keywords_entered = True
        st.session_state.awaiting_custom_keywords = False
        st.rerun()
elif st.session_state.custom_keywords_entered and not st.session_state.post_type:
    with st.chat_message("assistant"):
        st.markdown("### What type of content are you creating?")
    post_type = st.radio(
        "Select post type:",
        ["Blog Post", "LinkedIn Post", "Twitter Thread", "Email Newsletter"],
        index=None,
        label_visibility="collapsed"
    )
    if post_type:
        st.session_state.post_type = post_type
        st.session_state.messages.append({
            "role": "user",
            "content": f"Post Type: {post_type}"
        })
        st.rerun()

elif st.session_state.post_type and not st.session_state.tone and not st.session_state.awaiting_custom_tone:
    with st.chat_message("assistant"):
        st.markdown("### What tone would you like?")
    tone = st.radio(
        "Select tone:",
        ["Formal", "Professional", "Conversational", "Friendly", "Custom"],
        index=None,
        label_visibility="collapsed"
    )
    
    if tone == "Custom":
        st.session_state.awaiting_custom_tone = True
        st.rerun()
    elif tone:
        st.session_state.tone = tone
        st.session_state.messages.append({
            "role": "user",
            "content": f"Tone: {tone}"
        })
        st.rerun()

elif st.session_state.awaiting_custom_tone and not st.session_state.custom_tone:
    with st.chat_message("assistant"):
        st.markdown("Describe your custom tone:")
    custom_tone = st.text_input(
        "Custom Tone", 
        label_visibility="collapsed",
        placeholder="e.g., 'Academic but accessible', 'Playful but professional'"
    )
    
    if custom_tone:
        st.session_state.custom_tone = custom_tone
        st.session_state.tone = "Custom"
        st.session_state.awaiting_custom_tone = False
        st.session_state.messages.append({
            "role": "user",
            "content": f"Custom Tone: {custom_tone}"
        })
        st.rerun()

elif st.session_state.tone and not st.session_state.current_analysis:
    with st.spinner("Generating analysis with your keywords..."):
        if generate_analysis():
            st.rerun()

elif st.session_state.current_analysis and not st.session_state.article_generated:
    with st.spinner("Generating your custom article..."):
        if generate_article():
            st.rerun()

elif st.session_state.article_generated and not st.session_state.refinement_requested and not st.session_state.awaiting_refinement:
    with st.chat_message("assistant"):
        st.markdown("Would you like any improvements to the article?")
    refinement_choice = st.radio(
        "Improvement choice",
        ["Yes", "No"],
        index=None,
        label_visibility="collapsed"
    )
    
    if refinement_choice == "Yes":
        st.session_state.awaiting_refinement = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": "What improvements would you like? (Enter below and press Enter)"
        })
        st.rerun()
    elif refinement_choice == "No":
        st.session_state.refinement_requested = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Great! Your article is ready."
        })
        st.rerun()

elif st.session_state.awaiting_refinement and not st.session_state.refinement_requested:
    with st.chat_message("assistant"):
        st.markdown("Describe the improvements you'd like:")
    feedback_input = st.text_input(
        "Improvements", 
        label_visibility="collapsed",
        placeholder="e.g., Make it more concise, add more examples, etc."
    )
    
    if feedback_input:  
        st.session_state.messages.append({"role": "user", "content": feedback_input})
        with st.spinner("Making your requested changes..."):
            if refine_article(feedback_input):  
                st.session_state.awaiting_refinement = False
                st.session_state.refinement_requested = True
                st.rerun()

elif st.session_state.refinement_requested and not st.session_state.versions_generated:
    with st.spinner(f"Creating {st.session_state.post_type} content..."):
        versions = generate_post_with_human_version(
            post_type=st.session_state.post_type
        )

        if versions:
            st.session_state.versions_generated = True
            st.session_state.messages += [
                {
                    "role": "assistant",
                    "content": f"### 🤖 AI-Optimized {st.session_state.post_type}\n\n{versions['ai']}"
                },
                {
                    "role": "assistant",
                    "content": f"### ✍️ Humanized Version\n\n{versions['human']}"
                }
            ]
            st.rerun()

