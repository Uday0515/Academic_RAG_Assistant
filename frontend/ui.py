import streamlit as st
import pathlib
import datetime
import requests

API = "http://localhost:8000"

st.set_page_config(
    page_title="Dept. of Robotics & AI — Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #f7f8fc !important;
    font-family: 'Source Sans 3', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.dept-header {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a6b 60%, #0e2a52 100%);
    border-radius: 16px;
    padding: 28px 36px 22px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 24px rgba(10,22,40,0.18);
    position: relative;
    overflow: hidden;
}
.dept-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.dept-logo { font-size: 3rem; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3)); z-index: 1; }
.dept-text  { z-index: 1; }
.dept-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
    line-height: 1.2;
    margin: 0 0 4px 0;
}
.dept-subtitle {
    font-size: 0.82rem;
    color: #a8c4e8;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    font-weight: 400;
    margin: 0;
}
.dept-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    z-index: 1;
}
.dept-badge-label { font-size: 0.68rem; color: #a8c4e8; letter-spacing: 1.5px; text-transform: uppercase; display: block; }
.dept-badge-value { font-size: 1.1rem; font-weight: 600; color: #ffffff; }

[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #e8ecf4 !important;
    border-radius: 14px !important;
    padding: 4px 8px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 8px rgba(10,22,40,0.06) !important;
}
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 2px solid #c5d3ea !important;
    background: #ffffff !important;
    box-shadow: 0 2px 12px rgba(10,22,40,0.08) !important;
}
[data-testid="stChatInput"]:focus-within { border-color: #1a3a6b !important; }
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f4 !important;
}
.sidebar-section {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8899bb;
    font-weight: 600;
    margin: 18px 0 6px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #e8ecf4;
}
.stDownloadButton > button {
    background: #f0f4fb !important;
    border: 1px solid #c5d3ea !important;
    color: #1a3a6b !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #1a3a6b !important;
    color: #ffffff !important;
    border-color: #1a3a6b !important;
}
.welcome-card {
    background: #ffffff;
    border: 1px solid #e2e8f4;
    border-left: 4px solid #1a3a6b;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
    color: #3a4a6a;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(10,22,40,0.05);
}
.welcome-card strong { color: #0a1628; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()

doc_count = len(list(pathlib.Path("data").rglob("*.pdf"))) + len(st.session_state.uploaded_names)

st.markdown(f"""
<div class="dept-header">
    <div class="dept-logo">🤖</div>
    <div class="dept-text">
        <p class="dept-title">Department of Robotics and Artificial Intelligence</p>
        <p class="dept-subtitle">Academic Knowledge Assistant</p>
    </div>
    <div class="dept-badge">
        <span class="dept-badge-label">Documents</span>
        <span class="dept-badge-value">{doc_count}</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="sidebar-section">🎓 Institution</p>', unsafe_allow_html=True)
    st.markdown("**Dept. of Robotics & AI**")
    st.caption("Academic Knowledge Assistant")

    st.markdown('<p class="sidebar-section">📂 Filter by Subject</p>', unsafe_allow_html=True)
    try:
        subjects = requests.get(f"{API}/subjects").json().get("subjects", [])
    except:
        subjects = []
    options = ["All Subjects"] + subjects + (["Uploaded"] if st.session_state.uploaded_names else [])
    selected_subject = st.selectbox("Subject", options, label_visibility="collapsed")
    subject_filter = "All" if selected_subject == "All Subjects" else selected_subject

    st.markdown('<p class="sidebar-section">📤 Upload PDFs</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_names]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} PDF(s)…"):
                files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in new_files]
                requests.post(f"{API}/upload", files=files_payload)
                for f in new_files:
                    st.session_state.uploaded_names.add(f.name)
            st.success(f"✅ {len(new_files)} PDF(s) added!")

    if st.session_state.uploaded_names:
        with st.expander(f"📎 {len(st.session_state.uploaded_names)} uploaded file(s)"):
            for name in sorted(st.session_state.uploaded_names):
                st.caption(f"• {name}")

    st.markdown('<p class="sidebar-section">💾 Export Chat</p>', unsafe_allow_html=True)

    if st.session_state.messages:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        file_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        lines = [
            "DEPARTMENT OF ROBOTICS AND ARTIFICIAL INTELLIGENCE",
            "Academic Knowledge Assistant — Chat Export",
            f"Date: {timestamp}", "=" * 60, ""
        ]
        for msg in st.session_state.messages:
            label = "Student" if msg["role"] == "user" else "Assistant"
            lines += [f"{label}:", msg["content"], ""]

        st.download_button("⬇️ Download as .txt", data="\n".join(lines),
                           file_name=f"chat_{file_ts}.txt", mime="text/plain",
                           use_container_width=True)

        msg_html = ""
        for msg in st.session_state.messages:
            css = "user" if msg["role"] == "user" else "assistant"
            label = "Student" if msg["role"] == "user" else "Assistant"
            body = msg["content"].replace("\n", "<br>")
            msg_html += f'<div class="msg {css}"><strong>{label}:</strong><br>{body}</div>\n'

        html_out = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Chat Export</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;max-width:820px;margin:40px auto;padding:20px;background:#f7f8fc;color:#222}}
  h1{{color:#0a1628;font-size:1.3em}} p.sub{{color:#5570a0;font-size:.85em}}
  .msg{{padding:14px 18px;border-radius:10px;margin:10px 0;line-height:1.65}}
  .user{{background:#eef3fc;border-left:3px solid #1a3a6b}}
  .assistant{{background:#ffffff;border:1px solid #e2e8f4}}
</style></head><body>
<h1>Department of Robotics and Artificial Intelligence</h1>
<p class="sub">Academic Knowledge Assistant &nbsp;|&nbsp; {timestamp}</p><hr>
{msg_html}</body></html>"""

        st.download_button("⬇️ Download as HTML", data=html_out,
                           file_name=f"chat_{file_ts}.html", mime="text/html",
                           use_container_width=True)
    else:
        st.caption("No chat history yet.")

    st.markdown('<p class="sidebar-section">⚙️ Session</p>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <strong>Welcome to the Academic Knowledge Assistant</strong><br>
        Ask any question related to your course materials — syllabi, modules, question papers,
        or lecture notes. Use the <strong>subject filter</strong> in the sidebar to narrow results
        to a specific subject, or upload additional PDFs to expand the knowledge base.
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about your course materials…")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            response = requests.post(f"{API}/ask", json={
                "query": user_query,
                "subject_filter": subject_filter
            })
            data = response.json()
            answer = data["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})