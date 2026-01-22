import streamlit as st
from pathlib import Path
import tempfile
import os
import json
import time
from datetime import datetime
import logging
import traceback

# Import processors
from src.document_processors.pdf_processor import PDFProcessor
from src.document_processors.docx_processor import DOCXProcessor
from src.document_processors.pptx_processor import PPTXProcessor
from src.document_processors.xlsx_processor import XLSXProcessor

# Import knowledge base components
from src.knowledge_base.s3_manager import S3Manager
from src.knowledge_base.bedrock_kb import BedrockKnowledgeBase

# Import enhanced agent
from src.agents.enhanced_rag_agent import EnhancedRAGAgent

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize processors
PROCESSORS = {
    '.pdf': PDFProcessor(config.CHUNK_SIZES['pdf'], config.CHUNK_OVERLAPS['pdf']),
    '.docx': DOCXProcessor(config.CHUNK_SIZES['docx'], config.CHUNK_OVERLAPS['docx']),
    '.pptx': PPTXProcessor(config.CHUNK_SIZES['pptx'], config.CHUNK_OVERLAPS['pptx']),
    '.xlsx': XLSXProcessor(config.CHUNK_SIZES['xlsx'], config.CHUNK_OVERLAPS['xlsx'])
}

def main():
    st.set_page_config(
        page_title="Enhanced Multi-Format RAG Chatbot", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/rag-chatbot',
            'Report a bug': "https://github.com/your-repo/rag-chatbot/issues",
            'About': "# Enhanced RAG Chatbot\nBuilt with Streamlit, LangChain, and AWS Bedrock"
        }
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .status-info {
        background: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ff9800;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    
        🤖 Enhanced Multi-Format RAG Chatbot
        Upload PDF, DOCX, PPTX, XLSX files and chat with streaming responses, memory, and accuracy tracking!
    
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        render_sidebar()
    
    # Main content area
    if st.session_state.rag_agent:
        render_chat_interface()
    else:
        render_welcome_screen()

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'knowledge_base_id': None,
        'data_source_id': None,
        'rag_agent': None,
        'chat_messages': [],
        'processing_status': None,
        'kb_connected': False,
        'last_ingestion_job': None,
        'show_advanced_settings': False
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def render_sidebar():
    """Render the complete sidebar"""
    st.header("🎛️ Control Panel")
    
    # Knowledge Base Configuration
    render_kb_configuration()
    
    st.divider()
    
    # Document Upload Section
    render_document_upload()
    
    st.divider()
    
    # Memory Management (only if agent is connected)
    if st.session_state.rag_agent:
        render_memory_management()
        st.divider()
        render_accuracy_metrics()
    
    # Advanced Settings
    render_advanced_settings()

def render_kb_configuration():
    """Render Knowledge Base configuration section"""
    st.subheader("🗄️ Knowledge Base Setup")
    
    # Connection status
    if st.session_state.kb_connected:
        st.success("✅ Connected to Knowledge Base")
    else:
        st.info("ℹ️ Not connected to Knowledge Base")
    
    # Input fields
    kb_id = st.text_input(
        "Knowledge Base ID",
        value=st.session_state.knowledge_base_id or "",
        help="Enter your AWS Bedrock Knowledge Base ID (created via AWS Console)",
        placeholder="e.g., ABCD1234EFGH"
    )
    
    ds_id = st.text_input(
        "Data Source ID", 
        value=st.session_state.data_source_id or "",
        help="Enter your Data Source ID (created via AWS Console)",
        placeholder="e.g., WXYZ5678IJKL"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔗 Connect", type="primary", use_container_width=True):
            connect_to_knowledge_base(kb_id, ds_id)
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            if st.session_state.knowledge_base_id:
                refresh_kb_connection()
    
    # Show KB info if connected
    if st.session_state.kb_connected and st.session_state.knowledge_base_id:
        show_kb_info()

def connect_to_knowledge_base(kb_id: str, ds_id: str):
    """Connect to the knowledge base"""
    if not kb_id or not ds_id:
        st.error("❌ Please provide both Knowledge Base ID and Data Source ID")
        return
    
    try:
        with st.spinner("🔍 Validating Knowledge Base..."):
            kb_manager = BedrockKnowledgeBase()
            
            # Validate KB exists
            kb_info = kb_manager.get_knowledge_base(kb_id)
            
            # Validate data source exists
            data_sources = kb_manager.list_data_sources(kb_id)
            ds_exists = any(ds['dataSourceId'] == ds_id for ds in data_sources)
            
            if not ds_exists:
                st.error(f"❌ Data Source {ds_id} not found in Knowledge Base")
                return
            
            # Create RAG agent
            st.session_state.knowledge_base_id = kb_id
            st.session_state.data_source_id = ds_id
            st.session_state.rag_agent = EnhancedRAGAgent(
                knowledge_base_id=kb_id,
                memory_window=config.MEMORY_WINDOW_SIZE
            )
            st.session_state.kb_connected = True
            
            st.success(f"✅ Connected to KB: {kb_info.get('name', kb_id)}")
            logger.info(f"Connected to Knowledge Base: {kb_id}")
            st.rerun()
            
    except Exception as e:
        logger.error(f"Error connecting to KB: {e}")
        st.error(f"❌ Error connecting to KB: {str(e)}")
        st.session_state.kb_connected = False

def refresh_kb_connection():
    """Refresh the knowledge base connection"""
    try:
        with st.spinner("🔄 Refreshing connection..."):
            kb_manager = BedrockKnowledgeBase()
            kb_info = kb_manager.get_knowledge_base(st.session_state.knowledge_base_id)
            st.success("✅ Connection refreshed successfully")
    except Exception as e:
        st.error(f"❌ Error refreshing connection: {str(e)}")
        st.session_state.kb_connected = False

def show_kb_info():
    """Show knowledge base information"""
    try:
        kb_manager = BedrockKnowledgeBase()
        kb_info = kb_manager.get_knowledge_base(st.session_state.knowledge_base_id)
        
        with st.expander("📋 Knowledge Base Details", expanded=False):
            st.write(f"**Name:** {kb_info.get('name', 'N/A')}")
            st.write(f"**Status:** {kb_info.get('status', 'N/A')}")
            st.write(f"**Created:** {kb_info.get('createdAt', 'N/A')}")
            
            # Show data sources
            data_sources = kb_manager.list_data_sources(st.session_state.knowledge_base_id)
            st.write(f"**Data Sources:** {len(data_sources)}")
            
    except Exception as e:
        logger.warning(f"Could not fetch KB info: {e}")

def render_document_upload():
    """Render document upload section"""
    st.subheader("📤 Upload Documents")
    
    if not st.session_state.kb_connected:
        st.warning("⚠️ Please connect to a Knowledge Base first")
        return
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'pptx', 'xlsx'],
        accept_multiple_files=True,
        help="Upload multiple documents to add to your knowledge base"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected")
        
        # Show file details
        with st.expander("📋 File Details", expanded=False):
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024 / 1024  # MB
                st.write(f"• **{file.name}** ({file_size:.1f} MB)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents(uploaded_files)
        
        with col2:
            auto_ingest = st.checkbox("Auto-start ingestion", value=True, 
                                    help="Automatically start KB ingestion after upload")

def render_memory_management():
    """Render memory management section"""
    st.subheader("🧠 Memory Management")
    
    # Memory summary
    memory_info = st.session_state.rag_agent.get_memory_summary()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", memory_info['total_messages'])
    with col2:
        st.metric("Window Size", memory_info['memory_window'])
    
    st.caption(f"Session: {memory_info['session_id'][:8]}...")
    
    # Memory actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Memory", use_container_width=True):
            st.session_state.rag_agent.clear_memory()
            st.session_state.chat_messages = []
            st.success("Memory cleared!")
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            export_chat_history()

def export_chat_history():
    """Export chat history"""
    try:
        chat_data = st.session_state.rag_agent.export_chat_history()
        
        # Add metadata
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "knowledge_base_id": st.session_state.knowledge_base_id,
            "total_messages": len(chat_data),
            "chat_history": chat_data
        }
        
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error exporting chat: {str(e)}")

def render_accuracy_metrics():
    """Render accuracy metrics section"""
    st.subheader("📊 Accuracy Metrics")
    
    try:
        metrics = st.session_state.rag_agent.get_accuracy_report()
        
        # Main metrics in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Queries", metrics["total_queries"])
            confidence = metrics['average_confidence']
            confidence_delta = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
            st.metric("Avg Confidence", f"{confidence:.2f}", delta=confidence_delta)
        
        with col2:
            st.metric("Avg Retrieval", f"{metrics['average_retrieval_score']:.2f}")
            st.metric("Avg Time", f"{metrics['average_response_time']:.1f}s")
        
        # Additional metrics
        st.metric("Source Coverage", f"{metrics['source_coverage']:.2f}")
        
        # User satisfaction
        if metrics["total_feedback"] > 0:
            satisfaction = metrics['user_satisfaction']
            satisfaction_emoji = "😊" if satisfaction > 0.7 else "😐" if satisfaction > 0.4 else "😞"
            st.metric("User Satisfaction", f"{satisfaction:.2f}", delta=satisfaction_emoji)
            
            # Feedback breakdown
            st.markdown("**Feedback Breakdown:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"👍 **Positive:** {metrics['positive_feedback']}")
            with col2:
                st.markdown(f"👎 **Negative:** {metrics['negative_feedback']}")
        
        # Detailed metrics in expander
        with st.expander("📈 Detailed Analytics", expanded=False):
            st.json(metrics)
            
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")

def render_advanced_settings():
    """Render advanced settings section"""
    st.subheader("⚙️ Advanced Settings")
    
    if st.checkbox("Show Advanced Options", value=st.session_state.show_advanced_settings):
        st.session_state.show_advanced_settings = True
        
        # Retrieval settings
        st.markdown("**Retrieval Configuration:**")
        new_k = st.slider("Number of Retrieved Documents", 1, 10, config.RETRIEVAL_K)
        if new_k != config.RETRIEVAL_K:
            config.RETRIEVAL_K = new_k
            st.info(f"Retrieval K updated to {new_k}")
        
        # Memory settings
        st.markdown("**Memory Configuration:**")
        if st.session_state.rag_agent:
            current_window = st.session_state.rag_agent.memory.k
            new_window = st.slider("Memory Window Size", 1, 20, current_window)
            if new_window != current_window:
                st.session_state.rag_agent.memory.k = new_window
                st.info(f"Memory window updated to {new_window}")
        
        # Debug mode
        debug_mode = st.checkbox("Debug Mode", value=False)
        if debug_mode:
            st.markdown("**Debug Information:**")
            st.json({
                "session_state_keys": list(st.session_state.keys()),
                "config_values": {
                    "AWS_REGION": config.AWS_REGION,
                    "S3_BUCKET": config.S3_BUCKET,
                    "RETRIEVAL_K": config.RETRIEVAL_K
                }
            })
    else:
        st.session_state.show_advanced_settings = False

def render_chat_interface():
    """Render the main chat interface"""
    st.header("💬 Chat with Your Documents")
    
    # Chat container with custom styling
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_messages):
            render_chat_message(message, i)
    
    # Chat input
    render_chat_input()

def render_chat_message(message: dict, index: int):
    """Render a single chat message"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metrics for assistant messages
        if message["role"] == "assistant" and "metrics" in message:
            render_message_metrics(message["metrics"])
        
        # Show sources
        if message["role"] == "assistant" and message.get("sources"):
            render_message_sources(message["sources"])
        
        # Feedback buttons for assistant messages
        if message["role"] == "assistant" and "feedback_given" not in message:
            render_feedback_buttons(message, index)

def render_message_metrics(metrics: dict):
    """Render metrics for a message"""
    with st.expander("📈 Response Metrics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence = metrics.get('confidence_score', 0)
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            st.markdown(f"**Confidence:** {confidence:.2f}", 
                       unsafe_allow_html=True)
        
        with col2:
            retrieval = metrics.get('retrieval_score', 0)
            st.markdown(f"**Retrieval:** {retrieval:.2f}")
        
        with col3:
            response_time = metrics.get('response_time', 0)
            time_color = "green" if response_time < 5 else "orange" if response_time < 10 else "red"
            st.markdown(f"**Time:** {response_time:.1f}s", 
                       unsafe_allow_html=True)
        
        with col4:
            sources_count = metrics.get('sources_count', 0)
            st.markdown(f"**Sources:** {sources_count}")

def render_message_sources(sources: list):
    """Render sources for a message"""
    with st.expander("📚 Sources", expanded=False):
        for j, source in enumerate(sources, 1):
            st.markdown(f"**{j}.** {source}")

def render_feedback_buttons(message: dict, index: int):
    """Render feedback buttons for a message"""
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("👍", key=f"pos_{index}_{message.get('timestamp', '')}", 
                    help="This response was helpful"):
            st.session_state.rag_agent.add_user_feedback(True)
            message["feedback_given"] = "positive"
            st.success("Thanks for the positive feedback!")
            st.rerun()
    
    with col2:
        if st.button("👎", key=f"neg_{index}_{message.get('timestamp', '')}", 
                    help="This response was not helpful"):
            st.session_state.rag_agent.add_user_feedback(False)
            message["feedback_given"] = "negative"
            st.success("Thanks for the feedback! We'll work to improve.")
            st.rerun()

def render_chat_input():
    """Render chat input and handle responses"""
    # Sample questions
    with st.expander("💡 Sample Questions", expanded=False):
        sample_questions = [
            "What are the main topics covered in the documents?",
            "Can you summarize the key findings?",
            "What data is available in the spreadsheets?",
            "Are there any charts or images that show trends?",
            "What recommendations are mentioned?",
            "Compare information across different documents",
            "What are the most important insights?",
            "Show me specific data points or statistics"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                handle_user_input(question)
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents", key="main_chat_input"):
        handle_user_input(prompt)

def handle_user_input(prompt: str):
    """Handle user input and generate response"""
    # Add user message
    user_message = {
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate streaming response
    with st.chat_message("assistant"):
        generate_streaming_response(prompt)

def generate_streaming_response(prompt: str):
    """Generate and display streaming response"""
    response_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    sources_placeholder = st.empty()
    
    full_response = ""
    sources = []
    metrics = {}
    
    try:
        # Stream the response
        for chunk in st.session_state.rag_agent.stream_response(prompt):
            if chunk["type"] == "status":
                status_placeholder.markdown(f'{chunk["content"]}', 
                                          unsafe_allow_html=True)
            
            elif chunk["type"] == "content":
                full_response += chunk["content"]
                response_placeholder.markdown(full_response + "▌")
            
            elif chunk["type"] == "complete":
                full_response = chunk["content"]
                sources = chunk["sources"]
                metrics = chunk["metrics"]
                
                # Clear status and show final response
                status_placeholder.empty()
                response_placeholder.markdown(full_response)
                
                # Show metrics
                if metrics:
                    with metrics_placeholder.expander("📈 Response Metrics", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            confidence = metrics.get('confidence_score', 0)
                            st.metric("Confidence", f"{confidence:.2f}")
                        with col2:
                            retrieval = metrics.get('retrieval_score', 0)
                            st.metric("Retrieval", f"{retrieval:.2f}")
                        with col3:
                            response_time = metrics.get('response_time', 0)
                            st.metric("Time", f"{response_time:.1f}s")
                        with col4:
                            sources_count = metrics.get('sources_count', 0)
                            st.metric("Sources", sources_count)
                
                # Show sources
                if sources:
                    with sources_placeholder.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}.** {source}")
                
                break
            
            elif chunk["type"] == "error":
                status_placeholder.error(chunk["content"])
                full_response = "Sorry, I encountered an error while processing your question."
                response_placeholder.markdown(full_response)
                break
    
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        logger.error(traceback.format_exc())
        status_placeholder.error(f"Error: {str(e)}")
        full_response = "Sorry, I encountered an error while processing your question."
        response_placeholder.markdown(full_response)
    
    # Add assistant message to history
    assistant_message = {
        "role": "assistant",
        "content": full_response,
        "sources": sources,
        "metrics": metrics,
        "timestamp": datetime.now()
    }
    st.session_state.chat_messages.append(assistant_message)

def render_welcome_screen():
    """Render welcome screen when not connected"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        
            🚀 Welcome to Enhanced RAG Chatbot
            Get started by connecting to your Knowledge Base!
        
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Getting Started:")
        
        steps = [
            ("1️⃣", "**Create Knowledge Base**", "Set up your Knowledge Base in AWS Bedrock Console"),
            ("2️⃣", "**Create Data Source**", "Point your data source to your S3 bucket"),
            ("3️⃣", "**Enter Credentials**", "Input your KB ID and Data Source ID in the sidebar"),
            ("4️⃣", "**Upload Documents**", "Process your PDF, DOCX, PPTX, XLSX files"),
            ("5️⃣", "**Start Chatting**", "Ask questions and get intelligent responses!")
        ]
        
        for emoji, title, description in steps:
            st.markdown(f"""
            
                {emoji}
                
                    {title}
                    <small>{description}</small>
                
            
            """, unsafe_allow_html=True)
        
        st.markdown("### 💡 What You Can Ask:")
        
        sample_categories = {
            "📊 **Data Analysis**": [
                "What trends do you see in the data?",
                "Summarize the key statistics",
                "Compare different datasets"
            ],
            "📄 **Document Insights**": [
                "What are the main topics covered?",
                "Extract key recommendations",
                "Find specific information about..."
            ],
            "🖼️ **Visual Content**": [
                "Describe the charts and images",
                "What do the visualizations show?",
                "Extract text from images"
            ]
        }
        
        for category, questions in sample_categories.items():
            with st.expander(category, expanded=False):
                for question in questions:
                    st.markdown(f"• {question}")
        
        # Feature highlights
        st.markdown("### ✨ Key Features:")
        
        features = [
            ("🔄", "**Streaming Responses**", "Real-time token-by-token generation"),
            ("🧠", "**Conversation Memory**", "Context-aware multi-turn conversations"),
            ("📊", "**Accuracy Tracking**", "Confidence scores and performance metrics"),
            ("📁", "**Multi-Format Support**", "PDF, DOCX, PPTX, XLSX with image processing"),
            ("☁️", "**AWS Integration**", "Powered by Bedrock Knowledge Base"),
            ("👥", "**User Feedback**", "Rate responses to improve accuracy")
        ]
        
        cols = st.columns(2)
        for i, (emoji, title, description) in enumerate(features):
            with cols[i % 2]:
                st.markdown(f"""
                
                    {emoji} {title}
                    <small>{description}
                
                """, unsafe_allow_html=True)

def process_documents(uploaded_files):
    """Process uploaded documents and upload to S3 for ingestion"""
    try:
        # Create progress tracking
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_expander = st.expander("📋 Processing Details", expanded=True)
        
        with details_expander:
            processing_log = st.empty()
        
        log_messages = []
        
        def add_log(message):
            log_messages.append(f"• {message}")
            processing_log.markdown("\n".join(log_messages))
        
        add_log("🔄 Starting document processing...")
        all_chunks = []
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension not in PROCESSORS:
                add_log(f"❌ Unsupported file type: {file_extension} for {uploaded_file.name}")
                continue
            
            status_text.text(f"📄 Processing {uploaded_file.name}...")
            add_log(f"📄 Processing {uploaded_file.name} ({file_extension})...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = Path(tmp_file.name)
            
            try:
                # Process the file
                processor = PROCESSORS[file_extension]
                chunks = processor.extract_content(tmp_file_path)
                all_chunks.extend(chunks)
                
                add_log(f"✅ {uploaded_file.name}: {len(chunks)} chunks created")
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                add_log(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temp file
                if tmp_file_path.exists():
                    os.unlink(tmp_file_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files) * 0.7)  # 70% for processing
        
        # Upload to S3 and trigger ingestion
        if all_chunks:
            status_text.text("📤 Uploading to S3...")
            add_log(f"📤 Uploading {len(all_chunks)} chunks to S3...")
            
            try:
                s3_manager = S3Manager()
                kb_manager = BedrockKnowledgeBase()
                
                # Upload chunks to S3
                s3_uris = s3_manager.upload_chunks(all_chunks)
                progress_bar.progress(0.9)  # 90% after upload
                
                if s3_uris:
                    add_log(f"✅ Uploaded {len(s3_uris)} chunks to S3")
                    
                    # Start ingestion job
                    status_text.text("⚙️ Starting knowledge base ingestion...")
                    add_log("⚙️ Starting knowledge base ingestion job...")
                    
                    job_id = kb_manager.start_ingestion_job(
                        st.session_state.knowledge_base_id,
                        st.session_state.data_source_id
                    )
                    
                    st.session_state.last_ingestion_job = job_id
                    add_log(f"🔄 Ingestion job started: {job_id}")
                    
                    # Show ingestion status
                    ingestion_status = st.empty()
                    ingestion_status.info("⏳ Ingestion job is running. This may take a few minutes...")
                    
                    # Optional: Wait for ingestion with timeout
                    with st.spinner("Waiting for ingestion to complete..."):
                        success = kb_manager.wait_for_ingestion(
                            st.session_state.knowledge_base_id,
                            st.session_state.data_source_id,
                            job_id,
                            timeout=600  # 10 minutes
                        )
                        
                        if success:
                            add_log("🎉 Ingestion completed successfully!")
                            ingestion_status.success("🎉 Documents successfully ingested into knowledge base!")
                        else:
                            add_log("⚠️ Ingestion is taking longer than expected")
                            ingestion_status.warning("⚠️ Ingestion is taking longer than expected. Check AWS Console for status.")
                
                else:
                    add_log("❌ Failed to upload chunks to S3")
                    st.error("❌ Failed to upload chunks to S3")
                    
            except Exception as e:
                logger.error(f"Error in S3 upload or ingestion: {e}")
                add_log(f"❌ Error in upload/ingestion: {str(e)}")
                st.error(f"❌ Error in upload/ingestion: {str(e)}")
        
        else:
            add_log("❌ No chunks were created from the uploaded files")
            st.error("❌ No chunks were created from the uploaded files")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        add_log("✅ Document processing complete!")
        
        # Show summary
        if all_chunks:
            st.success(f"🎉 Successfully processed {len(uploaded_files)} files and created {len(all_chunks)} chunks!")
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        logger.error(traceback.format_exc())
        st.error(f"❌ Error processing documents: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Application error: {str(e)}")
        st.info("Please check the logs and try refreshing the page.")