import streamlit as st
import uuid
from PIL import Image
from report_qa_chat import ReportQASystem
from utils import get_analysis_by_id, get_latest_analyses
import os

api_key = os.getenv('GEMINI_API_KEY')
if "qa_system" not in st.session_state:
    st.session_state["qa_system"] = ReportQASystem(api_key=api_key)

def render_sidebar_qa():
    """Enhanced sidebar with proper session management"""
    st.sidebar.markdown("### 🩺 Report Q&A System")
    
    # User configuration
    if "qa_user_name" not in st.session_state:
        st.session_state["qa_user_name"] = "Patient"
    
    user_name = st.sidebar.text_input(
        "Your Name",
        value=st.session_state["qa_user_name"],
        key="qa_user_input"
    )
    st.session_state["qa_user_name"] = user_name
    
    # API key status
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if gemini_api_key:
        st.sidebar.success("✅ AI Assistant Ready")
        if "qa_system" not in st.session_state:
            st.session_state["qa_system"] = ReportQASystem(api_key=gemini_api_key)
    else:
        st.sidebar.error("⚠️ Gemini API key not found")
    
    st.sidebar.markdown("---")
    
    # Report selection with filename-based naming
    st.sidebar.markdown("### 📋 Select Report")
    latest_analyses = get_latest_analyses(limit=10)
    
    if latest_analyses:
        report_options = {}
        for analysis in latest_analyses:
            # Use filename for display if available
            filename = analysis.get('filename', 'Unknown')
            display_text = f"{analysis['date'][:10]} - {filename}"
            report_options[display_text] = analysis['id']
        
        selected_display = st.sidebar.selectbox(
            "Choose a report:",
            options=list(report_options.keys()),
            key="selected_report"
        )
        
        if selected_display:
            selected_report_id = report_options[selected_display]
            st.session_state["selected_report_id"] = selected_report_id
    else:
        st.sidebar.info("No reports available. Analyze some images first!")
    
    # QA Session management
    st.sidebar.markdown("### 💬 QA Sessions")
    
    if st.sidebar.button("🆕 New QA Session", use_container_width=True):
        if "qa_system" in st.session_state and gemini_api_key:
            # Use selected report filename for session naming
            session_name = "General Q&A"
            report_summary = "General medical questions"
            
            if "selected_report_id" in st.session_state:
                report = get_analysis_by_id(st.session_state["selected_report_id"])
                if report:
                    session_name = f"Q&A: {report.get('filename', 'Medical Report')}"
                    report_summary = report["analysis"]
                    report_id = report["id"]
            
            room_id = str(uuid.uuid4())[:8]
            qa_system = st.session_state["qa_system"]
            qa_system.create_qa_room(room_id, user_name, report_summary)
            
            # Store session name for display
            if "qa_session_names" not in st.session_state:
                st.session_state["qa_session_names"] = {}
            st.session_state["qa_session_names"][room_id] = session_name
            
            st.session_state["current_qa_room"] = room_id
            st.sidebar.success(f"Created: {session_name}")
            st.rerun()
    
    # Show existing QA sessions with proper navigation
    if "qa_system" in st.session_state:
        qa_system = st.session_state["qa_system"]
        existing_rooms = qa_system.get_qa_rooms()
        
        if existing_rooms:
            st.sidebar.markdown("**Active QA Sessions:**")
            for room in existing_rooms[-10:]:  # Show last 5
                # Get session name or use room ID
                session_name = st.session_state.get("qa_session_names", {}).get(
                    room['id'], 
                    f"Session {room['id']}"
                )
                
                if st.sidebar.button(
                    f"📋 {session_name}", 
                    key=f"join_qa_{room['id']}",
                    use_container_width=True
                ):
                    st.session_state["current_qa_room"] = room['id']
                    st.rerun()  # This will redirect to the chat window


def render_main_qa():
    if "current_qa_room" not in st.session_state:
        st.markdown("""
        <div class='report-card'>
            <h2>Welcome to the Report Q&A System</h2>
            <p>This tool helps you understand your medical reports better. You can ask questions in plain language to get clear explanations of complex terms and findings.</p>
            <strong>How to start:</strong>
            <ol>
                <li>Select a report from the sidebar.</li>
                <li>Click "Start New QA Session".</li>
                <li>Ask your questions in the chat window that appears.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    room_id = st.session_state["current_qa_room"]
    qa_system = st.session_state["qa_system"]
    room = next((r for r in qa_system.get_qa_rooms() if r["id"] == room_id), None)
    report_context = ""
    if room and room.get("report_summary"):
        report_context = room["report_summary"]
    st.markdown(f"### 💬 QA Session: `{room_id}`")
    
    messages_container = st.container(height=500)
    with messages_container:
        for msg in qa_system.get_messages(room_id):
            avatar = "🤖" if msg["user"] == "Medical AI Assistant" else "👤"
            with st.chat_message(msg["user"], avatar=avatar):
                st.markdown(msg["content"])
                if msg.get("image_attached"):
                    st.info("An image was considered with this query.")

    uploaded_image = st.file_uploader("Attach an image to your question", type=["jpg", "jpeg", "png"], key=f"qa_uploader_{room_id}")
    user_question = st.chat_input("Ask a question about your report...")

    if user_question:
        user_name = st.session_state.get("qa_user_name", "Patient")
        pil_image = Image.open(uploaded_image) if uploaded_image else None

        qa_system.add_message(room_id, user_name, user_question, image=bool(pil_image))

        # Always use the report_summary from the current room as context unless a selected_report_id_for_qa is set
        findings = []
        # Try to get full analysis if selected_report_id_for_qa is set
        if "selected_report_id_for_qa" in st.session_state:
            report = get_analysis_by_id(st.session_state["selected_report_id_for_qa"])
            if report:
                report_context = report["analysis"]
                findings = report.get("findings", [])
            else:
                # fallback to report_summary from room
                report_context = room["report_summary"] if room and room.get("report_summary") else ""
        else:
            report_context = room["report_summary"] if room and room.get("report_summary") else ""

        with st.spinner("AI is thinking..."):
            qa_system.answer_question(room_id, user_question, report_context, findings, image=pil_image)
            st.rerun()