import streamlit as st
import json
import os
import uuid
import time
from datetime import datetime
from PIL import Image
import io
import base64

def get_chat_store():
    """Get chat from storage"""
    try:
        if os.path.exists("chat_store.json"):
            with open("chat_store.json", "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        return {"rooms": {}}
    except Exception as e:
        print(f"Error loading chat store: {e}")
        return {"rooms": {}}

def save_chat_store(store):
    """Save the chat storage"""
    try:
        with open("chat_store.json", "w") as f:
            json.dump(store, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving chat store: {e}")

def create_chat_room(case_id, creator_name, case_description):
    """Create a new chat room for a case"""
    store = get_chat_store()

    if case_id not in store["rooms"]:
        room_data = {
            "id": case_id,
            "created_at": datetime.now().isoformat(),
            "creator": creator_name,
            "description": case_description,
            "participants": [creator_name, "Dr. AI Assistant", "Dr. Johnson", "Dr. Chen", "Dr. Patel"],
            "messages": []
        }

        # welcome_message = {
        #     "id": str(uuid.uuid4()),
        #     "user": "Dr. AI Assistant",
        #     "content": f"Welcome to the case discussion for '{case_description}'. I've analyzed the image and I'm here to assist with the diagnosis. Feel free to ask me specific questions about the findings or upload additional images for analysis.",
        #     "type": "text",
        #     "timestamp": datetime.now().isoformat()
        # }

        # room_data["messages"].append(welcome_message)
        store["rooms"][case_id] = room_data
        save_chat_store(store)

# Modified process_image_for_chat (in chat_system.py)
def process_image_for_chat(uploaded_file):
    """Process uploaded image for chat with Gemini Vision"""
    try:
        image = Image.open(uploaded_file) # The PIL Image object

        # Convert to base64 for storage
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image_base64": img_str,
            "filename": uploaded_file.name,
            "size": len(buffered.getvalue()),
            "image_pil": image # 🆕 CRUCIAL: Return the PIL object for AI analysis
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# In your chat_system.py, verify this function looks exactly like this, 
# especially the way the prompt is constructed:

# In chat_system.py

def analyze_image_with_gemini_vision(image, question, api_key, case_description=None, findings=None):
    """Analyze image using Gemini Vision model, accepting case context."""
    try:
        import google.generativeai as genai
        # ... (rest of imports and setup)
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-2.5-pro')

        # 🆕 Build Context Text
        context_text = f"Case Description: {case_description}\n"
        if findings and len(findings) > 0:
            context_text += "Key Findings Discussed: " + ", ".join(findings) + "\n"
        
        # 🆕 Update the prompt structure to include context text
        prompt = [
            image, # Ensure 'image' is a PIL.Image object
            f"""You are a medical AI assistant analyzing this image in a collaborative medical discussion.
            
            --- Case Context ---
            {context_text}
            --- End Context ---

            Question/Context: {question}

            Please provide:
            1. A detailed analysis of what you see in the image (e.g., this is a prescription/X-ray/lab result).
            2. Any medical findings or observations (e.g., drug name, dosage, pathology) in the context of the case.
            3. Answers to the specific question asked.
            4. Clinical insights that would be helpful for the medical team.

            Keep your response conversational but medically accurate, as this is part of a collaborative discussion."""
        ]

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def join_chat_room(case_id, user_name):
    """Join an existing chat room"""
    store = get_chat_store()
    if case_id in store["rooms"]:
        if user_name not in store["rooms"][case_id]["participants"]:
            store["rooms"][case_id]["participants"].append(user_name)
            save_chat_store(store)
        return True
    return False

# In chat_system.py

def add_message(case_id, user_name, message, message_type="text", image_data=None):
    """Add a message to a chat room, accepting optional image_data."""
    store = get_chat_store()
    if case_id in store["rooms"]:
        message_data = {
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            # 🆕 Add image fields
            "image_base64": image_data.get("image_base64") if image_data else None,
            "image_filename": image_data.get("filename") if image_data else None,
        }
        store["rooms"][case_id]["messages"].append(message_data)
        save_chat_store(store)
        return message_data
    return None

def get_messages(case_id, limit=50):
    """Get the most recent messages from a chat room"""
    store = get_chat_store()
    if case_id in store["rooms"]:
        messages = store["rooms"][case_id]["messages"]
        return messages[-limit:] if len(messages) > limit else messages
    return []

def get_available_rooms():
    """Get a list of all available chat rooms"""
    store = get_chat_store()
    rooms = []
    for room_id, room_data in store["rooms"].items():
        rooms.append({
            "id": room_id,
            "description": room_data["description"],
            "creator": room_data["creator"],
            "created_at": room_data["created_at"],
            "participants": len(room_data["participants"])
        })
    # Sort by creation date (newest first)
    rooms.sort(key=lambda x: x["created_at"], reverse=True)
    return rooms

def get_gemini_response(user_question, case_description, findings=None, api_key=None):
    """Get response from Gemini AI using the new google-genai library (inspired by your implementation)"""
    if not api_key:
        return "Please configure your Google Gemini API key to get AI responses."
    
    try:
        from google import genai
        from google.genai import types
        
        # Configure the client
        client = genai.Client(api_key=api_key)
        
        # Build findings text
        findings_text = ""
        if findings and len(findings) > 0:
            findings_text = "The key findings in the image are:\n"
            for i, finding in enumerate(findings, 1):
                findings_text += f"{i}. {finding}\n"

        # System prompt for medical AI assistant
        system_prompt = """You are DoctorAI, an advanced medical AI assistant specializing in radiology and clinical case discussions. You are participating in a collaborative chat room with clinicians to analyze and discuss medical cases.

Your role:
- Provide clear, structured, and evidence-based answers to user questions about the case and medical image.
- Use up-to-date clinical guidelines and literature when relevant.
- If information is missing or ambiguous, ask clarifying questions.
- Communicate with professionalism, empathy, and clinical accuracy.
- If uncertain, state your limitations and suggest next steps.
- Base your response on the findings and your medical expertise.
- Respond as if you are speaking directly to the doctor in a collaborative setting.
- Keep your response concise but informative, focusing on the relevant medical details."""

        # Combine case information with user question
        user_prompt = f"""Case Description: {case_description}

{findings_text}

Doctor's question: {user_question}"""

        # Configure generation parameters
        config = types.GenerateContentConfig(
            temperature=0.2,  # Lower temperature for medical accuracy
            top_p=0.9,
            max_output_tokens=6000,  # Sufficient for medical responses
        )
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.5-pro",  # Use available model
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=config
        )
        
        return response.text
        
    except ImportError:
        return "Google GenAI library not installed. Please install: pip install google-genai"
    except Exception as e:
        return f"AI response error: {str(e)}"

def generate_completion_with_continuation(prompt: str,
                                        model: str = "gemini-2.5-pro",
                                        system_prompt: str = "You are a helpful medical assistant.",
                                        **kwargs) -> str:
    """
    Generate completion with automatic continuation for long responses
    """
    from google import genai
    from google.genai import types
    
    # Configure the client
    client = genai.Client(api_key=kwargs.get('api_key'))
    
    # Combine system prompt and user prompt
    combined_prompt = f"{system_prompt}\n\nUser: {prompt}"
    
    # Configure generation parameters with higher token limit
    config = types.GenerateContentConfig(
    temperature=kwargs.get("temperature", 0.2),
    top_p=kwargs.get("top_p", 0.9),
    max_output_tokens=kwargs.get("max_output_tokens", 6000),  # match the name here
    candidate_count=1,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=combined_prompt,
            config=config
        )
        
        response_text = response.text
        
        # Check if response was cut off (common indicators)
        truncation_indicators = [
            "**",  # Markdown formatting left incomplete
            "###",  # Headers left incomplete
            "- ",   # List items cut off
            "\n\n"  # Abrupt ending with double newlines
        ]
        
        # If response ends abruptly, try to continue
        if any(response_text.rstrip().endswith(indicator.rstrip()) for indicator in truncation_indicators):
            continuation_prompt = f"Continue the previous response about: {prompt}\n\nPrevious response was:\n{response_text}\n\nPlease continue where it left off:"
            
            continuation_response = client.models.generate_content(
                model=model,
                contents=continuation_prompt,
                config=config
            )
            
            response_text += "\n" + continuation_response.text
        
        return response_text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Enhanced medical response function
def generate_medical_response(question: str, context: str = "", **kwargs) -> str:
    """
    Specialized function for medical responses with proper token management
    """
    medical_system_prompt = """You are an expert medical AI assistant providing comprehensive, accurate medical information. 

Guidelines:
- Provide complete, detailed responses without abrupt cutoffs
- Use clear medical terminology with explanations
- Structure responses with proper headings and sections
- If discussing treatment options, include all relevant categories
- Always complete your thoughts and recommendations fully
- End with clear next steps or questions for clarification

Ensure your response is complete and properly formatted."""

    return generate_completion_with_continuation(
        prompt=question,
        system_prompt=medical_system_prompt,
        model="gemini-2.5-pro",
        max_output_tokens=6000,  # Higher limit for medical responses
        temperature=0.2,   # Lower for accuracy
        **kwargs
    )

def create_manual_chat_room(case_id, creator_name, case_description, findings=None):
    """Create a chat room manually with findings"""
    store = get_chat_store()
    if case_id not in store["rooms"]:
        room_data = {
            "id": case_id,
            "created_at": datetime.now().isoformat(),
            "creator": creator_name,
            "description": case_description,
            "findings": findings or [],
            "participants": [creator_name, "Dr. AI Assistant"],
            "messages": []
        }
        
        # Create welcome message with findings
        findings_text = ""
        if findings:
            findings_text = f"\n\nKey findings identified:\n" + "\n".join([f"• {finding}" for finding in findings])
        
        welcome_message = {
            "id": str(uuid.uuid4()),
            "user": "Dr. AI Assistant",
            "content": f"Welcome to the case discussion for: {case_description}{findings_text}\n\nI'm here to assist with analysis and diagnosis. What would you like to discuss about this case?",
            "type": "text",
            "timestamp": datetime.now().isoformat()
        }
        room_data["messages"].append(welcome_message)
        store["rooms"][case_id] = room_data
        save_chat_store(store)
    return case_id

def render_chat_interface():
    """Render the enhanced chat interface with image upload support in the main chat area"""
    # Header section
    st.markdown("""
    <style>
    .custom-card {
        padding: 10px 20px;
        border-radius: 10px;
        background-color: #f0f2f6; /* Light gray background */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .custom-card h2 {
        color: #007bff; /* Blue for the heading */
        margin-top: 0;
    }
    </style>
    <div class="custom-card">
        <h2>🏥 Medical Collaboration Hub</h2>
        <p>Real-time case discussions with AI-powered insights and image analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Get API key from environment
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    # --- Sidebar for user configuration and room management ---
    with st.sidebar:
        st.markdown("### 👨‍⚕️ Doctor Profile")

        # User name input
        if "user_name" not in st.session_state:
            st.session_state.user_name = "Dr. Anonymous"

        user_name = st.text_input(
            "Your Name",
            value=st.session_state.user_name,
            placeholder="Enter your name (e.g., Dr. Smith)"
        )
        st.session_state.user_name = user_name

        # API key status (Modified: Removed st.text_input)
        st.markdown("### 🔑 AI Configuration")
        if gemini_api_key:
            st.success("✅ AI Assistant Ready (via GEMINI_API_KEY)")
        else:
            st.error("⚠️ Gemini API key not found in environment (GEMINI_API_KEY)")

        st.markdown("---")

        # Room management section
        st.markdown("### 🏠 Room Management")

        # Create new room
        with st.expander("➕ Create New Case Room", expanded=False):
            new_case_desc = st.text_area(
                "Case Description",
                placeholder="Brief description of the medical case...",
                max_chars=500
            )

            if st.button("🚀 Create Room", use_container_width=True):
                if new_case_desc and user_name:
                    case_id = str(uuid.uuid4())[:8]
                    create_chat_room(case_id, user_name, new_case_desc)
                    st.session_state.selected_room = case_id
                    st.success(f"✅ Room created: {case_id}")
                    st.rerun()
                else:
                    st.error("Please enter case description and your name")
        
        # Available rooms
        st.markdown("### 📋 Available Case Rooms")
        available_rooms = get_available_rooms()
        
        if available_rooms:
            for room in available_rooms[:5]:  # Show latest 5 rooms
                col1, col2 = st.columns([3, 1])
                with col1:
                    room_display = f"**{room['id']}**\n{room['description'][:50]}{'...' if len(room['description']) > 50 else ''}"
                    st.markdown(room_display)
                with col2:
                    if st.button("Join", key=f"join_{room['id']}", use_container_width=True):
                        join_chat_room(room['id'], user_name)
                        st.session_state.selected_room = room['id']
                        st.rerun()
                st.markdown(f"👥 {room['participants']} participants")
                st.markdown("---")
        else:
            st.info("No active case rooms. Create one to get started!")
            
        st.markdown("---")
        
        # NOTE: Removed the image uploader from the sidebar here.

    # --- Main chat interface ---
    if 'selected_room' not in st.session_state:
        # Welcome screen (unchanged)
        st.markdown("""
        ### 🎯 Welcome to Medical Collaboration Hub
        This platform enables real-time discussions between medical professionals with AI assistance for case analysis.
        **Get started:** Create a new case room or join an existing one from the sidebar.
        """)
        return

    # Chat interface for selected room
    room_id = st.session_state.selected_room
    
    # Chat header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 💬 Case Room: `{room_id}`")
    with col2:
        if st.button("🚪 Leave", help="Leave this chat room"):
            # Clear any pending image uploads upon leaving
            if 'image_to_send' in st.session_state: del st.session_state.image_to_send
            del st.session_state.selected_room
            st.rerun()

    # Get room info
    store = get_chat_store()
    if room_id not in store["rooms"]:
        st.error("Room not found!")
        return
    
    room_info = store["rooms"][room_id]
    
    # Show case description (unchanged)
    with st.expander("📋 Case Information", expanded=False):
        st.write(f"**Description:** {room_info['description']}")
        st.write(f"**Creator:** {room_info['creator']}")
        st.write(f"**Participants:** {', '.join(room_info.get('participants', []))}")
        if 'findings' in room_info and room_info['findings']:
            st.write("**Key Findings:**")
            for finding in room_info['findings']:
                st.write(f"• {finding}")

    # Chat messages container
    messages_container = st.container()
    
    # Get and display messages
    messages = get_messages(room_id)
    
    with messages_container:
        for msg in messages:
            avatar_key = "🤖" if msg["user"] == "Dr. AI Assistant" else "👤"
            with st.chat_message(msg["user"], avatar=avatar_key):
                # Display image if available (decode base64)
                # if msg.get("image_base64"):
                #     try:
                #         image_bytes = base64.b64decode(msg["image_base64"])
                #         image = Image.open(io.BytesIO(image_bytes))
                #         st.image(image, caption=f"Image uploaded by {msg['user']}", use_column_width=True)
                #     except Exception as e:
                #         st.warning(f"Could not display image: {str(e)}")
                if msg.get("image_filename"):
                    st.markdown(f"**[🖼️ Image Attached: {msg['image_filename']}]**")
                st.markdown(f"**{msg['user']}:** {msg['content']}")
                
    # --- Chat input and logic (NOW USING st.form) ---
    with st.form(key='chat_form', clear_on_submit=True):
        
        # 🆕 Image Uploader placed above text input, mimicking the reference image style
        uploaded_file_obj = st.file_uploader(
            "Attach an image to your question (Scan, X-ray, Photo)", 
            type=["jpg", "jpeg", "png"],
            key="chat_image_uploader",
            help="Upload an image to include with your message for AI analysis."
        )

        user_message = st.text_input(
            "Ask a question about your report...",
            key="user_text_input",
            label_visibility="collapsed" # Hide the label for a cleaner chat interface
        )
        
        # Send button is now the form submit button
        submit_button = st.form_submit_button(label='Send', use_container_width=True)
        
    if submit_button and (user_message or uploaded_file_obj):
        
        image_to_send = None
        if uploaded_file_obj:
            # Process and store the image data
            processed_data = process_image_for_chat(uploaded_file_obj)
            if processed_data:
                image_to_send = processed_data
            else:
                st.error("Could not process image. Sending text only.")

        # Ensure message has content before proceeding
        if not user_message and image_to_send:
             # If only an image is uploaded, use a default prompt
            user_message = "Please analyze this image and provide initial findings."
        elif not user_message and not image_to_send:
            # Should not happen with current logic, but a safeguard
            st.warning("Please type a message or upload an image.")
            return

        # Add user message
        add_message(room_id, user_name, user_message, image_data=image_to_send)
        
        # Generate AI response if API key is provided and AI is a participant
        if gemini_api_key and "Dr. AI Assistant" in room_info.get('participants', []):
            with st.spinner("🤖 AI is analyzing..."):
                try:
                    findings = room_info.get('findings', [])
                    
                    if image_to_send and image_to_send.get('image_pil'):
                        # Multi-modal request
                        ai_response = analyze_image_with_gemini_vision(
                            image=image_to_send['image_pil'],
                            question=user_message,
                            case_description=room_info['description'],
                            findings=findings,
                            api_key=gemini_api_key
                        )
                    else:
                        # Text-only request
                        ai_response = get_gemini_response(
                            user_message, 
                            room_info['description'], 
                            findings=findings,
                            api_key=gemini_api_key
                        )
                        
                    add_message(room_id, "Dr. AI Assistant", ai_response)
                    
                except Exception as e:
                    st.error(f"⚠️ AI response error: {str(e)}")
        
        # Rerun to clear the form/update the chat history
        st.rerun()

    # Quick action buttons (adapted to include image logic)
    st.markdown("---")
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    quick_questions = [
        "What's your differential diagnosis?",
        "What additional tests do you recommend?",
        "Can you explain the key findings in the image?"
    ]
    
    for i, (col, question) in enumerate(zip([col1, col2, col3], quick_questions)):
        with col:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                # For quick actions, we check if an image was last sent
                # Note: We can't use the form uploader state directly here,
                # we'd need to check the history, but for simplicity, we 
                # assume the user wants the AI to analyze the most recent image/context
                
                # We won't try to send a new image with a quick action button press,
                # as that should be done via the chat input form.
                image_to_send = None # Quick action buttons only send text prompt

                # Add the quick question as a message
                add_message(room_id, user_name, question) # No image data for quick prompts
                
                # Generate AI response if API key is provided
                if gemini_api_key and "Dr. AI Assistant" in room_info.get('participants', []):
                    with st.spinner("🤖 AI is analyzing..."):
                        try:
                            findings = room_info.get('findings', [])
                            
                            # Use text-only AI function for quick buttons
                            ai_response = get_gemini_response(
                                question, 
                                room_info['description'], 
                                findings=findings,
                                api_key=gemini_api_key
                            )
                                
                            add_message(room_id, "Dr. AI Assistant", ai_response)
                            
                        except Exception as e:
                            st.error(f"⚠️ AI response error: {str(e)}")

                st.rerun()


