import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import io
import re
from typing import Dict, Any, List, Union
from datetime import datetime
# Ensure pypdf and python-docx are installed: pip install pypdf python-docx
import pypdf 
import docx 

# ============================================================
# ➡️ GLOBAL TYPE ALIASES & CONSTANTS
# ============================================================

# Define the custom types (using base types for stability)
StandardAverages = Dict[str, float]
SentenceResult = Dict[str, Any]
DocumentResult = Dict[str, Any]

# Paul's Universal Intellectual Standards (Assumed Global/Imported)
PAUL_STANDARDS = {
    "clarity": {"name": "Clarity", "icon": "🔍", "color": "#1abc9c", "question": "Could you elaborate further on that point?"},
    "accuracy": {"name": "Accuracy", "icon": "✅", "color": "#2ecc71", "question": "How could we check on the truth of that statement?"},
    "precision": {"name": "Precision", "icon": "🎯", "color": "#3498db", "question": "Could you be more specific?"},
    "relevance": {"name": "Relevance", "icon": "📌", "color": "#9b59b6", "question": "How does that relate to the issue?"},
    "depth": {"name": "Depth", "icon": "🧱", "color": "#f1c40f", "question": "What factors make this a difficult problem?"},
    "breadth": {"name": "Breadth", "icon": "🌍", "color": "#e67e22", "question": "Do we need to look at this from another perspective?"},
    "logic": {"name": "Logic", "icon": "💡", "color": "#e74c3c", "question": "Does this make sense? How is it connected?"},
    "significance": {"name": "Significance", "icon": "⭐", "color": "#c0392b", "question": "Is this the most important problem to consider?"},
    "fairness": {"name": "Fairness", "icon": "⚖️", "color": "#34495e", "question": "Am I considering all relevant viewpoints in good faith?"},
}

SCORE_LEVELS = {
    "excellent": {"label": "Excellent", "min": 0.80, "color": "#2ECC71", "icon": "🌟"},
    "good": {"label": "Good", "min": 0.65, "color": "#3498DB", "icon": "👍"},
    "adequate": {"label": "Adequate", "min": 0.50, "color": "#F1C40F", "icon": "⚠️"},
    "needs_work": {"label": "Needs Work", "min": 0.0, "color": "#E74C3C", "icon": "❌"},
}

# ============================================================
# ➡️ HELPER FUNCTIONS AND STUBS
# ============================================================

class CriticalThinkingAnalyzer:
    """
    Mocks the complex logic for analyzing text against critical thinking standards.
    """
    def __init__(self):
        self.standards = PAUL_STANDARDS
        self.score_levels = SCORE_LEVELS
        self.standard_keys = list(PAUL_STANDARDS.keys())

    def analyze_document(self, sentences: List[str], doc_name: str, doc_id: str) -> DocumentResult:
        if not sentences:
            return {"total_sentences": 0, "standard_std_dev": 0.0, "overall_average": 0.0, "overall_level": SCORE_LEVELS["needs_work"]}

        # Simple mock logic for demonstration purposes
        
        # Base mock score slightly below average to show "Needs Work" level
        base_mock_score = 0.45 
        
        standard_scores_list = []
        sentence_overall_scores = []
        sentence_results = []

        for i, sentence in enumerate(sentences):
            # Simulate slight variation across standards
            standards_analysis = {}
            sentence_standards_scores = []
            
            for k in self.standard_keys:
                # Simulate a score that varies slightly from the base mock score
                score = max(0.0, min(1.0, base_mock_score + np.random.uniform(-0.15, 0.15)))
                
                standards_analysis[k] = {
                    "standard_name": PAUL_STANDARDS[k]["name"],
                    "score": score,
                    "level": self._get_score_level(score),
                    "color": PAUL_STANDARDS[k]["color"],
                    "icon": PAUL_STANDARDS[k]["icon"],
                    "feedback": f"Mock feedback for {PAUL_STANDARDS[k]['name'].lower()}. This standard scored {score:.0%}.",
                    "question": PAUL_STANDARDS[k]['question']
                }
                sentence_standards_scores.append(score)
            
            # Sentence overall score is the average of its standard scores
            sentence_overall_score = np.mean(sentence_standards_scores)
            sentence_overall_scores.append(sentence_overall_score)
            
            level_data = self._get_score_level(sentence_overall_score)
            standard_scores_list.append(sentence_standards_scores) # List of lists for correlation
            
            sentence_results.append({
                "index": i + 1,
                "sentence": sentence,
                "word_count": len(sentence.split()),
                "standards": standards_analysis,
                "overall_score": sentence_overall_score,
                "overall_level": level_data,
                "strengths": [PAUL_STANDARDS["clarity"]["name"]],
                "weaknesses": [PAUL_STANDARDS["fairness"]["name"]],
                "recommendations": []
            })
        
        # Calculate final document averages and statistics
        df_scores = pd.DataFrame(standard_scores_list, columns=self.standard_keys)
        
        standard_averages = df_scores.mean().to_dict()
        overall_avg = np.mean(sentence_overall_scores)
        overall_level = self._get_score_level(overall_avg)
        
        # ➡️ ENHANCEMENT 1: Consistency Metrics
        standard_std_dev = df_scores.mean().std() # Std Dev of the 9 average standard scores
        
        return {
            "document_name": doc_name,
            "document_id": doc_id,
            "total_sentences": len(sentences),
            "sentence_results": sentence_results,
            "standard_averages": standard_averages,
            "overall_average": overall_avg,
            "overall_level": overall_level,
            "standard_std_dev": standard_std_dev,
            "df_scores": df_scores # Keep DataFrame for correlation calculation
        }

    def _get_score_level(self, score: float) -> Dict[str, Any]:
        """Finds the appropriate level (Excellent, Good, etc.) for a given score."""
        for level_data in self.score_levels.values():
            if score >= level_data["min"]:
                return level_data
        return self.score_levels["needs_work"]


def extract_text_from_file(file_source: Union[io.BytesIO, Any], file_type: str) -> str:
    """Extracts text from various file types."""
    if file_type == 'pdf':
        try:
            reader = pypdf.PdfReader(file_source)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception:
            return "ERROR_PDF_EXTRACTION"
    elif file_type == 'docx':
        try:
            document = docx.Document(file_source)
            return "\n".join(p.text for p in document.paragraphs if p.text)
        except Exception:
            return "ERROR_DOCX_EXTRACTION"
    elif file_type == 'txt':
        try:
            return file_source.read().decode('utf-8')
        except Exception:
            return "ERROR_TXT_EXTRACTION"
    else:
        return f"ERROR_UNSUPPORTED_TYPE"


def preprocess_text(text: str) -> List[str]:
    """Splits text into sentences and filters."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    # Filter out very short or empty strings
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


# ============================================================
# ➡️ VISUALIZATION FUNCTIONS
# ============================================================

def create_standard_radar(averages: StandardAverages, title: str) -> go.Figure:
    """Generates a Plotly radar chart for standards comparison."""
    categories = [PAUL_STANDARDS[k]["name"] for k in averages.keys()]
    values = [averages[k] for k in averages.keys()]
    r_values = values + [values[0]]
    theta_categories = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r_values, theta=theta_categories, fill='toself', fillcolor='rgba(102, 126, 234, 0.3)', line=dict(color='#667eea', width=2), name='Score'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0], tickfont=dict(size=10)), angularaxis=dict(tickfont=dict(size=11))), title=title, height=450, template="plotly_dark", margin=dict(t=50, b=50, l=70, r=70))
    return fig


def create_standards_bar_chart(averages: StandardAverages, title: str) -> go.Figure:
    """Generates a Plotly horizontal bar chart for standards ranking."""
    data = []
    color_map = {}
    for key, score in averages.items():
        standard_info = PAUL_STANDARDS[key]
        data.append({"Standard": standard_info["name"], "Score": score})
        color_map[standard_info["name"]] = standard_info["color"]
    df = pd.DataFrame(data).sort_values("Score", ascending=True)
    fig = px.bar(df, x="Score", y="Standard", orientation='h', color="Standard", color_discrete_map=color_map)
    fig.update_layout(title=title, xaxis=dict(range=[0, 1], title="Score"), yaxis=dict(title=""), height=400, template="plotly_dark", showlegend=False)
    return fig


def create_progress_chart(sentence_results: List[SentenceResult], title: str) -> go.Figure:
    """Generates a Plotly line chart showing score progression."""
    if not sentence_results: return go.Figure()
    indices = [r["index"] for r in sentence_results]
    overall_scores = [r["overall_score"] for r in sentence_results]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=overall_scores, mode='lines+markers', name='Overall Score', line=dict(color='#667eea', width=3), marker=dict(size=5)))
    x_np, y_np = np.array(indices), np.array(overall_scores)
    
    # Calculate trend line
    if len(x_np) > 1:
        z = np.polyfit(x_np, y_np, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=x_np, y=p(x_np), mode='lines', name=f'Trend Line (Slope: {z[0]:.3f})', line=dict(color='#E74C3C', width=2, dash='dash')))
    
    fig.update_layout(title=title, xaxis_title="Sentence Number", yaxis_title="Score", yaxis=dict(range=[0, 1.05]), height=350, template="plotly_dark", hovermode="x unified")
    return fig


def create_sentence_heatmap(sentence_results: List[SentenceResult], title: str) -> go.Figure:
    """Generates a Plotly heatmap of scores per sentence and standard."""
    if not sentence_results: return go.Figure()
    standards_keys = list(PAUL_STANDARDS.keys())
    # Limit to first 30 sentences for readability
    matrix = [[result["standards"][s]["score"] for s in standards_keys] for result in sentence_results[:30]]
    labels = [f"S{result['index']}" for result in sentence_results[:30]]
    x_labels = [PAUL_STANDARDS[s]["name"] for s in standards_keys]
    fig = px.imshow(matrix, x=x_labels, y=labels, color_continuous_scale="RdYlGn", aspect="auto", title=title)
    fig.update_layout(height=max(400, len(labels) * 25), template="plotly_dark", xaxis=dict(tickangle=45), coloraxis_colorbar=dict(title="Score"))
    return fig

# ➡️ ENHANCEMENT 3: Correlation Heatmap
def create_standards_correlation_heatmap(df_scores: pd.DataFrame, title: str) -> go.Figure:
    """Generates a Plotly heatmap showing the correlation matrix between all standards."""
    if df_scores.empty or len(df_scores) < 2: return go.Figure()
    
    # Calculate the Pearson correlation matrix
    correlation_matrix = df_scores.corr()
    standards_names = [PAUL_STANDARDS[key]['name'] for key in df_scores.columns]
    
    fig = px.imshow(
        correlation_matrix, 
        x=standards_names, 
        y=standards_names, 
        color_continuous_scale="RdBu", # Use a diverging color scale for correlation
        zmin=-1, zmax=1,
        title=title,
        text_auto=".2f",
        aspect="auto"
    )
    
    fig.update_layout(
        template="plotly_dark", 
        height=500,
        xaxis=dict(tickangle=45),
        coloraxis_colorbar=dict(title="Correlation")
    )
    return fig


# ============================================================
# ➡️ HTML GENERATION FUNCTIONS
# ============================================================

def generate_sentence_report_html(result: SentenceResult) -> str:
    """Generates the HTML content for a single sentence analysis."""
    # (The HTML generation function is retained for the download button functionality)
    overall_level = result['overall_level']
    html = f"""<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid {overall_level['color']};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0; color: white;">📝 Sentence {result['index']} (Word Count: {result['word_count']})</h4>
            <span style="background: {overall_level['color']}; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; font-size: 14px;">
                {overall_level['icon']} {overall_level['label']} ({result['overall_score']:.0%})
            </span>
        </div>
        <p style="color: #ccc; font-style: italic; margin-bottom: 20px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            "{result['sentence']}"
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">"""
    for analysis in result["standards"].values():
        level_color = analysis["level"]["color"]
        feedback_summary = analysis['feedback'][:90] + ('...' if len(analysis['feedback']) > 90 else '')
        html += f"""
            <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; border-top: 3px solid {analysis['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {analysis['color']}; font-weight: bold; font-size: 13px;">{analysis['icon']} {analysis['standard_name']}</span>
                    <span style="background: {level_color}; padding: 2px 8px; border-radius: 10px; font-size: 12px; color: white;">{analysis['score']:.0%}</span>
                </div>
                <p style="color: #999; font-size: 11px; margin: 8px 0 0 0; line-height: 1.4;">{feedback_summary}</p>
            </div>"""
    html += "</div>"
    if result["strengths"] or result["weaknesses"]:
        html += '<div style="display: flex; gap: 20px; margin-top: 15px;">'
        if result["strengths"]: html += f"""<div style="flex: 1; background: rgba(46, 204, 113, 0.1); padding: 10px; border-radius: 8px; border: 1px solid #2ECC71;"><strong style="color: #2ECC71;">💪 Strengths:</strong><span style="color: #ccc;"> {', '.join(result['strengths'])}</span></div>"""
        if result["weaknesses"]: html += f"""<div style="flex: 1; background: rgba(231, 76, 60, 0.1); padding: 10px; border-radius: 8px; border: 1px solid #E74C3C;"><strong style="color: #E74C3C;">🎯 Weaknesses:</strong><span style="color: #ccc;"> {', '.join(result['weaknesses'])}</span></div>"""
        html += "</div>"
    html += "</div>"
    return html


def get_level_data(score: float, score_levels: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to find the level for a score based on SCORE_LEVELS dictionary."""
    for level_data in score_levels.values():
        if score >= level_data["min"]:
            return level_data
    return score_levels["needs_work"]


def generate_full_report_html(doc_result: DocumentResult) -> str:
    """ 
    Generates a complete, downloadable HTML report including document summary and 
    all detailed sentence analyses.
    """
    strongest_standard = max(doc_result['standard_averages'], key=doc_result['standard_averages'].get)
    weakest_standard = min(doc_result['standard_averages'], key=doc_result['standard_averages'].get)

    # --- HTML Head and Styles ---
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Critical Thinking Analysis Report - {doc_result['document_name']}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f0f23; color: #fff; padding: 40px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; }}
            h1, h2, h3 {{ color: white; margin-top: 0; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
            .metric {{ background: #1a1a2e; padding: 20px; border-radius: 10px; text-align: center; border-left: 3px solid #667eea;}}
            .metric-value {{ font-size: 2.2em; font-weight: bold; color: #667eea; }}
            .metric-label {{ color: #888; font-size: 0.9em; margin-top: 5px; }}
            .score-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; color: white; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; font-size: 0.95em;}}
            th {{ background: #282c3f; color: #ccc; }}
            tr:hover {{ background: #16213e; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 Critical Thinking Analysis Report</h1>
            <p>Based on Paul's Universal Intellectual Standards</p>
            <h3>Document: {doc_result['document_name']} | Total Sentences: {doc_result['total_sentences']}</h3>
            <p style="font-size: 0.9em;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>📈 Document Summary</h2>
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-value" style="color: {doc_result['overall_level']['color']};">{doc_result['overall_average']:.0%}</div>
                <div class="metric-label">Overall Average Score</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: {doc_result['overall_level']['color']};">{doc_result['overall_level']['icon']}</div>
                <div class="metric-label">Overall Level: {doc_result['overall_level']['label']}</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: {PAUL_STANDARDS[strongest_standard]['color']};">{PAUL_STANDARDS[strongest_standard]['icon']}</div>
                <div class="metric-label">Strongest: {PAUL_STANDARDS[strongest_standard]['name']}</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: {PAUL_STANDARDS[weakest_standard]['color']};">{PAUL_STANDARDS[weakest_standard]['icon']}</div>
                <div class="metric-label">Weakest: {PAUL_STANDARDS[weakest_standard]['name']}</div>
            </div>
        </div>
        
        <h2>📊 Standards Performance Table</h2>
        <table>
            <tr>
                <th>Standard</th>
                <th>Score</th>
                <th>Level</th>
                <th>Focus Question</th>
            </tr>
    """
    
    # --- Standard Averages Table ---
    for key, score in doc_result['standard_averages'].items():
        std = PAUL_STANDARDS[key]
        level_data = get_level_data(score, SCORE_LEVELS)
        
        html += f"""
            <tr>
                <td style="color: {std['color']}; font-weight: bold;">{std['icon']} {std['name']}</td>
                <td>{score:.0%}</td>
                <td><span class="score-badge" style="background: {level_data['color']};">{level_data['label']}</span></td>
                <td style="color: #888; font-style: italic;">{std['question']}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>📝 Detailed Sentence Analysis</h2>
        <p style="color: #aaa;">Review the breakdown for each sentence below.</p>
    """
    
    # --- Detailed Sentence Reports (reusing the first function) ---
    for result in doc_result['sentence_results']:
        html += generate_sentence_report_html(result)
    
    html += """
    </body>
    </html>
    """
    return html


# ============================================================
# ➡️ MAIN STREAMLIT APP
# ============================================================

def main():
    """
    Main function for the Streamlit Critical Thinking Analyzer application.
    Sets up the layout, handles user input (text/file), runs the analysis, 
    and displays the results using visualizations and HTML reports.
    """
    st.set_page_config(layout="wide", page_title="Paul's Critical Thinking Analyzer", page_icon="🧠")
    
    # Custom CSS for dark theme and metrics cards
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.8); margin: 10px 0 0 0; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        height: 100%; /* Ensure all cards have same height */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-value-large {
        font-size: 3em;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.2em;
    }
    .metric-label-small {
        color: #888;
        font-size: 0.9em;
        line-height: 1.2;
    }
    .sidebar-header {
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stCodeBlock {
        background-color: #1a1a2e !important;
        border: 1px solid #333;
    }
    .dataframe-container th {
        background-color: #282c3f !important;
        color: #ccc !important;
    }
    .dataframe-container td {
        background-color: #1a1a2e !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main Title Header
    st.markdown('<div class="main-header"><h1>🧠 Paul\'s Critical Thinking Analyzer</h1><p>Assess text against Paul\'s Universal Intellectual Standards (Clarity, Accuracy, Precision, etc.)</p></div>', unsafe_allow_html=True)

    # --- Sidebar for Input ---
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>Input Text</h2></div>', unsafe_allow_html=True)

        input_method = st.radio("Choose Input Method", ["Text Input", "File Upload"], index=0)

        uploaded_file = None
        user_text = ""
        doc_name = "User Input Text"

        if input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload Document (.txt, .pdf, .docx)", type=['txt', 'pdf', 'docx'])
            if uploaded_file:
                doc_name = uploaded_file.name
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                uploaded_file.seek(0)
                text_result = extract_text_from_file(uploaded_file, file_type)
                
                if text_result.startswith("ERROR_"):
                    st.error(f"File extraction failed. Ensure required libraries (pypdf, python-docx) are installed.")
                else:
                    user_text = text_result
        else:
            default_text = "The current economic policy is obviously flawed. It should be changed immediately because everyone knows a different system will clearly yield better results, but all the politicians are too corrupt to understand the simple solution."
            user_text = st.text_area("Paste Text for Analysis", height=300, value=default_text)
            
        st.subheader("Raw Text Preview")
        st.code(user_text[:500] + ('...' if len(user_text) > 500 else ''), language='text')

    # --- Main Content Area ---
    if user_text:
        try:
            sentences = preprocess_text(user_text)
            if not sentences:
                st.warning("The extracted text is too short or could not be properly segmented into sentences (min 10 characters).")
                st.stop()

            analyzer = CriticalThinkingAnalyzer()
            doc_id = str(hash(user_text))
            doc_result = analyzer.analyze_document(sentences, doc_name, doc_id)

            st.success(f"Analysis Complete: {doc_result['total_sentences']} sentences processed from **{doc_name}**.")
            
            # --- 1. Overall Metrics ---
            st.header("📊 Document Overview")
            col1, col2, col3, col4, col5 = st.columns(5) # Added one more column for the new metric

            overall_level = doc_result['overall_level']
            strongest = max(doc_result['standard_averages'], key=doc_result['standard_averages'].get)
            weakest = min(doc_result['standard_averages'], key=doc_result['standard_averages'].get)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large" style="color: {overall_level['color']};">{overall_level['icon']}</div>
                    <div class="metric-label-small">Overall Level: <strong>{overall_level['label']}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large">{doc_result['overall_average']:.1%}</div>
                    <div class="metric-label-small">Average Critical Thinking Score</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large" style="color: {PAUL_STANDARDS[strongest]['color']};">{PAUL_STANDARDS[strongest]['icon']}</div>
                    <div class="metric-label-small">Strongest Standard: <strong>{PAUL_STANDARDS[strongest]['name']}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large" style="color: {PAUL_STANDARDS[weakest]['color']};">{PAUL_STANDARDS[weakest]['icon']}</div>
                    <div class="metric-label-small">Weakest Standard: <strong>{PAUL_STANDARDS[weakest]['name']}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # ➡️ ENHANCEMENT 1: Display Consistency Metric
            with col5:
                # Lower Std Dev is better (more consistent)
                consistency_color = "#2ECC71" if doc_result['standard_std_dev'] < 0.05 else "#F1C40F"
                consistency_icon = "👍" if doc_result['standard_std_dev'] < 0.05 else "⚠️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large" style="color: {consistency_color};">{doc_result['standard_std_dev']:.3f}</div>
                    <div class="metric-label-small">Standard Deviation of Scores (Consistency) {consistency_icon}</div>
                </div>
                """, unsafe_allow_html=True)


            st.markdown("---")

            # --- 2. Visualization Charts ---
            st.header("📈 Visualization of Standards")
            
            # Row 1 of Charts (Radar, Bar)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(create_standard_radar(doc_result['standard_averages'], "Standards Radar Chart"), use_container_width=True)
            with chart_col2:
                st.plotly_chart(create_standards_bar_chart(doc_result['standard_averages'], "Average Score by Standard"), use_container_width=True)

            # Row 2 of Charts (Progress, Heatmap/Correlation)
            st.header("🔬 Advanced Analysis")
            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                # Progress Chart: Time series analysis
                st.plotly_chart(create_progress_chart(doc_result['sentence_results'], "Critical Thinking Score Progression Through Document"), use_container_width=True)
                
            with chart_col4:
                # ➡️ ENHANCEMENT 3: Correlation Heatmap
                if doc_result['total_sentences'] > 1:
                    st.plotly_chart(create_standards_correlation_heatmap(doc_result['df_scores'], "Standards Score Correlation Matrix"), use_container_width=True)
                    
                else:
                    st.info("Need more than one sentence to calculate standards correlation.")


            if len(doc_result['sentence_results']) > 1:
                st.plotly_chart(create_sentence_heatmap(doc_result['sentence_results'], "Sentence-by-Sentence Score Heatmap (First 30)"), use_container_width=True)

            st.markdown("---")

            # --- 3. Detailed Sentence Breakdown ---
            st.header("📝 Detailed Sentence Breakdown")
            
            # Selector for Sentence
            sentence_options = {r['index']: r['sentence'][:70] + '...' for r in doc_result['sentence_results']}
            selected_index = st.selectbox("Select a Sentence to View Details", 
                                          options=list(sentence_options.keys()), 
                                          format_func=lambda x: f"Sentence {x}: {sentence_options[x]}",
                                          key="sentence_selector")

            selected_result = next((r for r in doc_result['sentence_results'] if r['index'] == selected_index), None)

            # ➡️ ENHANCEMENT 4: Display feedback in a native Streamlit table
            if selected_result:
                st.subheader(f"Analysis for Sentence {selected_index}")
                st.markdown(f"**Original Text:** *\"{selected_result['sentence']}\"*")
                st.markdown(f"**Overall Score:** **{selected_result['overall_score']:.1%}** ({selected_result['overall_level']['label']} {selected_result['overall_level']['icon']})")
                
                # Prepare data for native DataFrame
                standards_data = []
                for key, analysis in selected_result['standards'].items():
                    standards_data.append({
                        "Standard": f"{analysis['icon']} {analysis['standard_name']}",
                        "Score": f"{analysis['score']:.1%}",
                        "Level": analysis['level']['label'],
                        "Feedback & Question": f"{analysis['feedback']} | Focus: {analysis['question']}"
                    })
                
                df_detail = pd.DataFrame(standards_data)

                # Use markdown table for better styling control or dataframe
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(
                    df_detail, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Standard": st.column_config.Column(width="small"),
                        "Score": st.column_config.Column(width="small"),
                        "Level": st.column_config.Column(width="small"),
                        "Feedback & Question": st.column_config.Column(width="large"),
                    }
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Strengths/Weaknesses summary
                col_s, col_w = st.columns(2)
                with col_s:
                    st.success(f"💪 **Strengths:** {', '.join(selected_result['strengths'])}")
                with col_w:
                    st.error(f"🎯 **Weaknesses:** {', '.join(selected_result['weaknesses'])}")


            st.markdown("---")

            # --- 4. Download Report ---
            st.header("⬇️ Download Full Report")
            full_html_report = generate_full_report_html(doc_result)
            
            st.download_button(
                label="Download Full HTML Report",
                data=full_html_report,
                file_name=f"{doc_name.replace(' ', '_')}_critical_thinking_report.html",
                mime="text/html"
            )


        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}. Check console for details.")
            st.exception(e)

    else:
        st.info("Paste or upload a document in the sidebar to begin the Critical Thinking Analysis.")

if __name__ == '__main__':
    # Add a check for essential non-standard libraries (pypdf, docx) if you want robust error handling before main()
    # E.g., try: import pypdf except ImportError: print("Pypdf not installed. File upload will fail.")
    main()
