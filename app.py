# critical_thinking_analyzer.py
# Analyzes text using Paul's Standards of Critical Thinking

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import torch
import pypdf
import docx
import re
from datetime import datetime
from typing import List, Dict, Any, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path

# ============================================================
#       PAUL'S STANDARDS OF CRITICAL THINKING - DEFINITIONS
# ============================================================

PAUL_STANDARDS = {
    "clarity": {
        "name": "Clarity",
        "color": "#3498DB",
        "icon": "🔍",
        "description": "Is the statement clear and understandable?",
        "question": "Could you elaborate? Could you illustrate? Could you give an example?",
        "indicators_positive": ["specifically", "for example", "in other words", "to illustrate", "meaning", "that is", "namely", "such as", "defined as", "to clarify"],
        "indicators_negative": ["somehow", "something", "stuff", "things", "whatever", "kind of", "sort of", "like", "you know", "etc"],
        "weight": 1.0
    },
    "accuracy": {
        "name": "Accuracy",
        "color": "#2ECC71",
        "icon": "✓",
        "description": "Is the statement true and free from errors?",
        "question": "How could we verify this? How could we find out if this is true?",
        "indicators_positive": ["according to", "research shows", "data indicates", "evidence suggests", "studies confirm", "verified", "documented", "proven", "factually", "statistics show"],
        "indicators_negative": ["everyone knows", "obviously", "clearly", "always", "never", "all", "none", "definitely", "absolutely certain", "no doubt"],
        "weight": 1.0
    },
    "precision": {
        "name": "Precision",
        "color": "#9B59B6",
        "icon": "🎯",
        "description": "Is the statement specific and detailed enough?",
        "question": "Could you be more specific? Could you give more details?",
        "indicators_positive": ["exactly", "precisely", "approximately", "measured", "calculated", "percent", "ratio", "specifically", "in particular", "detailed"],
        "indicators_negative": ["a lot", "many", "few", "some", "often", "sometimes", "rarely", "big", "small", "good", "bad", "nice", "very"],
        "weight": 1.0
    },
    "relevance": {
        "name": "Relevance",
        "color": "#E67E22",
        "icon": "🔗",
        "description": "Does the statement relate to the issue at hand?",
        "question": "How does this relate to the problem? How does this help with the issue?",
        "indicators_positive": ["therefore", "consequently", "as a result", "this relates to", "connected to", "relevant because", "pertinent", "applicable", "bearing on", "in relation to"],
        "indicators_negative": ["by the way", "incidentally", "speaking of", "anyway", "besides", "also", "moreover", "furthermore", "in addition"],
        "weight": 1.0
    },
    "depth": {
        "name": "Depth",
        "color": "#E74C3C",
        "icon": "📊",
        "description": "Does the statement address the complexity of the issue?",
        "question": "What factors make this difficult? What are the complexities?",
        "indicators_positive": ["underlying", "fundamental", "root cause", "complexity", "nuanced", "multifaceted", "layers", "deeper", "systematic", "comprehensive", "thorough"],
        "indicators_negative": ["simple", "easy", "just", "only", "merely", "basic", "straightforward", "obvious solution"],
        "weight": 1.0
    },
    "breadth": {
        "name": "Breadth",
        "color": "#1ABC9C",
        "icon": "🌐",
        "description": "Does the statement consider other viewpoints?",
        "question": "Is there another way to look at this? What would this look like from another perspective?",
        "indicators_positive": ["alternatively", "on the other hand", "from another perspective", "considering also", "however", "conversely", "different view", "opposing argument", "some argue", "others believe"],
        "indicators_negative": ["the only way", "must be", "has to be", "no other", "single solution", "one answer"],
        "weight": 1.0
    },
    "logic": {
        "name": "Logic",
        "color": "#F1C40F",
        "icon": "⚙️",
        "description": "Does the statement make sense and follow logically?",
        "question": "Does this follow from the evidence? Does this really make sense together?",
        "indicators_positive": ["because", "therefore", "thus", "hence", "consequently", "it follows that", "logically", "reasoning", "if then", "implies", "leads to"],
        "indicators_negative": ["but", "although", "despite", "regardless", "anyway", "still"],
        "weight": 1.0
    },
    "significance": {
        "name": "Significance",
        "color": "#8E44AD",
        "icon": "⭐",
        "description": "Is this the most important issue to focus on?",
        "question": "Is this the most important problem to consider? Which of these facts is most important?",
        "indicators_positive": ["importantly", "significantly", "crucially", "essentially", "fundamentally", "key point", "primary", "central", "critical", "vital", "paramount"],
        "indicators_negative": ["trivial", "minor", "insignificant", "unimportant", "negligible"],
        "weight": 1.0
    },
    "fairness": {
        "name": "Fairness",
        "color": "#16A085",
        "icon": "⚖️",
        "description": "Is the statement free from bias and self-interest?",
        "question": "Is my thinking justifiable? Am I considering others' viewpoints sympathetically?",
        "indicators_positive": ["objectively", "impartially", "fairly", "balanced", "unbiased", "neutral", "considering all", "without prejudice", "equitably", "justly"],
        "indicators_negative": ["obviously wrong", "stupid", "idiotic", "ridiculous", "absurd", "they always", "those people", "typical"],
        "weight": 1.0
    }
}

SCORE_LEVELS = {
    "excellent": {"min": 0.75, "color": "#2ECC71", "label": "Excellent", "icon": "🌟"},
    "good": {"min": 0.55, "color": "#3498DB", "label": "Good", "icon": "✅"},
    "adequate": {"min": 0.35, "color": "#F1C40F", "label": "Adequate", "icon": "⚠️"},
    "needs_work": {"min": 0.0, "color": "#E74C3C", "label": "Needs Improvement", "icon": "❌"}
}
# ============================================================
#               DATA EXTRACTION & PREPROCESSING
# ============================================================

def extract_text_from_file(file_source: Union[str, Path, io.BytesIO], file_type: str) -> str:
    """
    Extracts text content from PDF, DOCX, or TXT files.

    Args:
        file_source: The path to the file (str or Path) or a BytesIO object.
        file_type: The file extension ('pdf', 'docx', 'txt').

    Returns:
        The extracted text as a single, cleaned string, or an ERROR message.
    """
    text = ""
    file_type = file_type.strip().lower()

    try:
        match file_type:
            case 'pdf':
                # pypdf can handle both file paths and byte streams
                reader = pypdf.PdfReader(file_source)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            case 'docx':
                # docx.Document can handle file paths and file-like objects
                document = docx.Document(file_source)
                text = "\n".join(p.text for p in document.paragraphs if p.text)
            
            case 'txt':
                if isinstance(file_source, (str, Path)):
                    # Read from file path
                    text = Path(file_source).read_text(encoding='utf-8')
                elif isinstance(file_source, io.BytesIO):
                    # Decode content from bytes stream
                    text = file_source.read().decode('utf-8')
                else:
                    return f"ERROR_TXT_SOURCE: Unsupported source type for TXT file: {type(file_source)}"
            
            case _:
                return f"ERROR_UNSUPPORTED_TYPE: {file_type}"

    except FileNotFoundError:
        return "ERROR_FILE_NOT_FOUND"
    except pypdf.errors.PdfReadError as e:
        return f"ERROR_PDF_EXTRACTION: Invalid PDF file or password protected. Details: {e}"
    except Exception as e:
        # Catch other potential errors (docx corruption, decoding issues, etc.)
        return f"ERROR_EXTRACTION_FAILED: {e.__class__.__name__}: {e}"
    
    # Clean the extracted text: replace multiple whitespaces with single space and strip
    return " ".join(text.split()).strip()

# ---

def preprocess_text(text: str) -> List[str]:
    """
    Splits the cleaned text into sentences, filtering out short or empty strings.

    Args:
        text: The cleaned text string.

    Returns:
        A list of cleaned sentences (strings).
    """
    # Regex splits by periods, question marks, or exclamation points followed by one or more spaces
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    # Filter for non-empty sentences longer than 10 characters
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
# ============================================================
#         PAUL'S CRITICAL THINKING ANALYZER ENGINE
# ============================================================

class CriticalThinkingAnalyzer:
    """
    Analyzes text against Richard Paul's Standards of Critical Thinking
    using indicator keywords and structural heuristics.
    """
    def __init__(self, standards: Dict[str, Any], score_levels: Dict[str, Any]):
        """Initializes the analyzer with defined standards and scoring levels."""
        self.standards = standards
        self.score_levels = score_levels
        self.standard_keys = list(standards.keys())

    def _get_score_level(self, score: float) -> Dict[str, Any]:
        """Determine the performance level based on score."""
        for level_data in self.score_levels.values():
            if score >= level_data["min"]:
                return level_data
        # Default to the lowest level if score is below all defined minimums
        return self.score_levels["needs_work"]
    
    # --- Core Analysis Methods ---

    def _calculate_base_score(self, standard: Dict[str, Any], sentence_lower: str) -> tuple[float, List[str], List[str]]:
        """Calculates the score adjustment based purely on keyword indicators."""
        positive_indicators = standard.get("indicators_positive", [])
        negative_indicators = standard.get("indicators_negative", [])
        
        found_positive = [ind for ind in positive_indicators if ind.lower() in sentence_lower]
        found_negative = [ind for ind in negative_indicators if ind.lower() in sentence_lower]

        # Start neutral (0.5)
        base_score = 0.5 
        
        # Adjust for positive indicators (max +0.4)
        positive_count = len(found_positive)
        base_score += min(positive_count * 0.15, 0.4)
        
        # Penalize for negative indicators (max -0.35)
        negative_count = len(found_negative)
        base_score -= min(negative_count * 0.12, 0.35)
        
        return base_score, found_positive, found_negative

    def _apply_heuristics(self, sentence: str, standard_key: str, words: List[str]) -> float:
        """Applies additional structural and content-based score adjustments."""
        adjustment = 0.0
        sentence_lower = sentence.lower()
        word_count = len(words)
        
        match standard_key:
            case "clarity":
                if 8 < word_count < 30: adjustment += 0.05
                if "?" in sentence: adjustment += 0.05
                if word_count < 5: adjustment -= 0.1
                
            case "accuracy":
                if any(char.isdigit() for char in sentence): adjustment += 0.1
                if any(q in sentence for q in ['"', "'"]): adjustment += 0.05 # Quotes suggest citation
                
            case "precision":
                digit_count = sum(1 for c in sentence if c.isdigit())
                adjustment += min(digit_count * 0.03, 0.15)
                if "%" in sentence or "percent" in sentence_lower: adjustment += 0.1
                
            case "relevance":
                connectors = ["this", "that", "which", "these", "those"]
                if any(c in words for c in connectors): adjustment += 0.05
                
            case "depth":
                if word_count > 15: adjustment += 0.08
                if sentence.count(",") >= 2: adjustment += 0.05 # Multiple clauses
                
            case "breadth":
                comparatives = ["while", "whereas", "compared", "contrast", "both", "either"]
                if any(c in sentence_lower for c in comparatives): adjustment += 0.1
                
            case "logic":
                causal = ["cause", "effect", "result", "lead", "due to", "since"]
                if any(c in sentence_lower for c in causal): adjustment += 0.1
                
            case "significance":
                emphasis = ["must", "need", "essential", "require", "necessary"]
                if any(e in sentence_lower for e in emphasis): adjustment += 0.08
                
            case "fairness":
                if "we" in words or "our" in words: adjustment += 0.05 # First person plural
                # Absolute language reduces fairness
                absolutes = ["always", "never", "everyone", "no one", "all", "none"]
                if any(a in words for a in absolutes): adjustment -= 0.1
                
        return adjustment

    def _generate_feedback(self, standard_key: str, score: float, 
                           found_positive: List[str], found_negative: List[str]) -> str:
        """Generates constructive feedback based on the standard and score."""
        standard = self.standards[standard_key]
        name = standard["name"].lower()
        question = standard["question"]
        
        if score >= 0.75:
            base = f"Excellent {name}! "
            if found_positive:
                base += f"Good use of strong indicators like: {', '.join(found_positive[:3])}."
        elif score >= 0.55:
            base = f"Good {name}. "
            if found_negative:
                base += f"Consider replacing vague or absolute terms like: {', '.join(found_negative[:2])}."
            else:
                base += "Could strengthen the argument with more explicit, specific language."
        elif score >= 0.35:
            base = f"Adequate {name}, but needs improvement. "
            base += f"Focus on this question: {question}"
        else: # Needs Improvement
            base = f"{standard['name']} needs significant improvement. "
            if found_negative:
                base += f"Avoid vague or absolute terms like: {', '.join(found_negative[:2])}. "
            base += f"Ask yourself: {question}"
            
        return base
    
    def analyze_standard(self, sentence: str, standard_key: str) -> Dict[str, Any]:
        """Analyzes a sentence against a single Critical Thinking Standard."""
        standard = self.standards[standard_key]
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        # 1. Calculate base score from indicators
        base_score, found_positive, found_negative = self._calculate_base_score(standard, sentence_lower)
        
        # 2. Apply structural heuristics
        score_adjustment = self._apply_heuristics(sentence, standard_key, words)
        final_score = base_score + score_adjustment
        
        # 3. Clamp score between 0 and 1
        final_score = max(0.0, min(1.0, final_score))
        
        # 4. Generate results
        level = self._get_score_level(final_score)
        feedback = self._generate_feedback(standard_key, final_score, found_positive, found_negative)
        
        return {
            "standard_key": standard_key,
            "standard_name": standard["name"],
            "score": final_score,
            "level": level,
            "positive_indicators": found_positive,
            "negative_indicators": found_negative,
            "feedback": feedback,
            "question": standard["question"]
        }

    # --- Document Analysis Methods ---

    def analyze_sentence(self, sentence: str, index: int) -> Dict[str, Any]:
        """Analyzes a sentence against all Critical Thinking Standards and provides an overall assessment."""
        results = {
            "index": index,
            "sentence": sentence,
            "word_count": len(sentence.split()),
            "standards": {},
            "overall_score": 0.0,
            "overall_level": None,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        total_score = 0.0
        all_analyses = []
        
        for standard_key in self.standard_keys:
            analysis = self.analyze_standard(sentence, standard_key)
            results["standards"][standard_key] = analysis
            total_score += analysis["score"]
            all_analyses.append((standard_key, analysis["score"]))
            
        # Calculate overall score and level
        results["overall_score"] = total_score / len(self.standards)
        results["overall_level"] = self._get_score_level(results["overall_score"])
        
        # Identify top 3 strengths and bottom 3 weaknesses (if below threshold)
        sorted_analyses = sorted(all_analyses, key=lambda x: x[1], reverse=True)
        
        results["strengths"] = [self.standards[s[0]]["name"] for s in sorted_analyses[:3] if s[1] >= self.score_levels["good"]["min"]]
        results["weaknesses"] = [self.standards[s[0]]["name"] for s in sorted_analyses[-3:] if s[1] < self.score_levels["adequate"]["min"]]
        
        # Generate recommendations from the lowest scoring standards
        for standard_key, score in sorted_analyses[-2:]:
            if score < self.score_levels["adequate"]["min"]:
                results["recommendations"].append(self.standards[standard_key]["question"])
                
        return results
        
    def analyze_document(self, sentences: List[str], doc_name: str = "Document", doc_id: str = "0") -> Dict[str, Any]:
        """Analyzes an entire document (list of sentences) and provides document-level statistics."""
        sentence_results = []
        standard_totals = {k: 0.0 for k in self.standards}
        
        for i, sentence in enumerate(sentences):
            # Pass document metadata to sentence result (optional but useful)
            result = self.analyze_sentence(sentence, i + 1)
            result["document_name"] = doc_name
            result["document_id"] = doc_id
            sentence_results.append(result)
            
            # Accumulate scores for document average
            for standard_key in self.standard_keys:
                standard_totals[standard_key] += result["standards"][standard_key]["score"]
        
        # Calculate document-level statistics
        num_sentences = len(sentences)
        if num_sentences == 0:
             return { "document_name": doc_name, "document_id": doc_id, "error": "No sentences to analyze." }

        standard_averages = {k: v / num_sentences for k, v in standard_totals.items()}
        overall_avg = sum(standard_averages.values()) / len(standard_averages)
        
        return {
            "document_name": doc_name,
            "document_id": doc_id,
            "total_sentences": num_sentences,
            "sentence_results": sentence_results,
            "standard_averages": standard_averages,
            "overall_average": overall_avg,
            "overall_level": self._get_score_level(overall_avg)
        }
# ============================================================
#               VISUALIZATION FUNCTIONS (Plotly)
# ============================================================

def create_standard_radar(averages: Dict[str, float], title: str) -> go.Figure:
    """
    Creates a radar chart to visualize the overall score for each Critical Thinking Standard.
    """
    # Unpack standards data and align lists
    categories = [PAUL_STANDARDS[k]["name"] for k in averages.keys()]
    values = [averages[k] for k in categories] # Ensure order matches categories
    
    # Close the loop for the radar chart
    r_values = values + [values[0]]
    theta_categories = categories + [categories[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_values,
        theta=theta_categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1], 
                tickvals=[0.25, 0.5, 0.75, 1.0], 
                tickfont=dict(size=10) # Added tickfont here for consistency
            ),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        title=title,
        height=450,
        template="plotly_dark",
        margin=dict(t=50, b=50, l=70, r=70) # Add margins
    )
    return fig

# ---

def create_standards_bar_chart(averages: Dict[str, float], title: str) -> go.Figure:
    """
    Creates a horizontal bar chart showing the average score for each standard,
    colored by the standard's defined color.
    """
    data = []
    color_map = {}
    
    for key, score in averages.items():
        standard_info = PAUL_STANDARDS[key]
        data.append({
            "Standard": standard_info["name"],
            "Score": score,
            "Color": standard_info["color"],
            "Icon": standard_info["icon"]
        })
        color_map[standard_info["name"]] = standard_info["color"]
    
    df = pd.DataFrame(data)
    df = df.sort_values("Score", ascending=True) # Sort for better visual comparison
    
    fig = px.bar(
        df, 
        x="Score", 
        y="Standard", 
        orientation='h',
        color="Standard", # Color based on Standard name
        color_discrete_map=color_map # Use the defined color map
    )
    
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1], title="Score"),
        yaxis=dict(title=""),
        height=400,
        template="plotly_dark",
        showlegend=False
    )
    return fig

# ---

def create_sentence_heatmap(sentence_results: List[Dict[str, Any]], title: str) -> go.Figure:
    """
    Creates a heatmap showing the score of every sentence against every standard.
    (Limited to the first 30 sentences for readability.)
    """
    if not sentence_results:
        return go.Figure().add_annotation(text="No sentence data available.")
    
    standards_keys = list(PAUL_STANDARDS.keys())
    
    # Limit data for matrix to the first 30 sentences
    matrix = [
        [result["standards"][s]["score"] for s in standards_keys]
        for result in sentence_results[:30]
    ]
    labels = [f"S{result['index']}" for result in sentence_results[:30]]
    
    x_labels = [PAUL_STANDARDS[s]["name"] for s in standards_keys]
    
    fig = px.imshow(
        matrix,
        x=x_labels,
        y=labels,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title=title
    )
    
    fig.update_layout(
        height=max(400, len(labels) * 25), # Dynamically adjust height
        template="plotly_dark",
        xaxis=dict(tickangle=45),
        coloraxis_colorbar=dict(title="Score")
    )
    return fig

# ---

def create_score_distribution(sentence_results: List[SentenceResult], title: str) -> go.Figure:
    """
    Creates a histogram showing the distribution of the overall scores across all sentences.
    """
    scores = [r["overall_score"] for r in sentence_results]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='#667eea',
        opacity=0.8,
        name='Score Distribution'
    ))
    
    # Add threshold lines using SCORE_LEVELS
    for level_key, level_data in SCORE_LEVELS.items():
        if level_data["min"] > 0:
            fig.add_vline(
                x=level_data["min"], 
                line=dict(dash="dash", color=level_data["color"], width=1.5),
                annotation_text=level_data["label"],
                annotation_position="top" # Place text above the line
            )
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="Overall Score", range=[0, 1.05]), # Extend range slightly
        yaxis_title="Number of Sentences",
        height=350,
        template="plotly_dark",
        bargap=0.01 # Reduce gap between bars
    )
    return fig

# ---

def create_progress_chart(sentence_results: List[SentenceResult], title: str) -> go.Figure:
    """
    Creates a line chart showing the progression of the overall score through the document,
    including a linear trend line.
    """
    if not sentence_results:
        return go.Figure().add_annotation(text="No sentence data available.")
    
    indices = [r["index"] for r in sentence_results]
    overall_scores = [r["overall_score"] for r in sentence_results]
    
    fig = go.Figure()
    
    # Overall score line with markers
    fig.add_trace(go.Scatter(
        x=indices,
        y=overall_scores,
        mode='lines+markers',
        name='Overall Score',
        line=dict(color='#667eea', width=3),
        marker=dict(size=5)
    ))
    
    # Add linear trend line (requires numpy)
    x_np = np.array(indices)
    y_np = np.array(overall_scores)
    
    # Calculate linear regression (y = mx + c)
    z = np.polyfit(x_np, y_np, 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=x_np,
        y=p(x_np),
        mode='lines',
        name=f'Trend Line (Slope: {z[0]:.3f})',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sentence Number",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        height=350,
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

# ============================================================
#               REPORT GENERATION FUNCTIONS
# ============================================================

def generate_sentence_report_html(result: SentenceResult) -> str:
    """
    Generates a stylized HTML block report for a single sentence analysis result.
    """
    overall_level = result['overall_level']
    
    # --- Start HTML Structure ---
    html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 15px; margin: 15px 0; 
                border-left: 5px solid {overall_level['color']};">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0; color: white;">📝 Sentence {result['index']} (Word Count: {result['word_count']})</h4>
            <span style="background: {overall_level['color']}; padding: 5px 15px; 
                         border-radius: 20px; color: white; font-weight: bold; font-size: 14px;">
                {overall_level['icon']} {overall_level['label']} ({result['overall_score']:.0%})
            </span>
        </div>
        
        <p style="color: #ccc; font-style: italic; margin-bottom: 20px; 
                  padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            "{result['sentence']}"
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
    """
    
    # --- Standard Score Grid ---
    for analysis in result["standards"].values():
        level_color = analysis["level"]["color"]
        # Truncate feedback to prevent card explosion
        feedback_summary = analysis['feedback'][:90]
        if len(analysis['feedback']) > 90:
            feedback_summary += '...'
            
        html += f"""
            <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px;
                         border-top: 3px solid {analysis['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {analysis['color']}; font-weight: bold; font-size: 13px;">
                        {analysis['icon']} {analysis['standard_name']}
                    </span>
                    <span style="background: {level_color}; padding: 2px 8px; border-radius: 10px; 
                                 font-size: 12px; color: white;">
                        {analysis['score']:.0%}
                    </span>
                </div>
                <p style="color: #999; font-size: 11px; margin: 8px 0 0 0; line-height: 1.4;">
                    {feedback_summary}
                </p>
            </div>
        """
    
    html += "</div>"
    
    # --- Strengths and Weaknesses Block ---
    if result["strengths"] or result["weaknesses"]:
        html += '<div style="display: flex; gap: 20px; margin-top: 15px;">'
        
        if result["strengths"]:
            html += f"""
                <div style="flex: 1; background: rgba(46, 204, 113, 0.1); padding: 10px; border-radius: 8px; border: 1px solid #2ECC71;">
                    <strong style="color: #2ECC71;">💪 Strengths:</strong>
                    <span style="color: #ccc;"> {', '.join(result['strengths'])}</span>
                </div>
            """
            
        if result["weaknesses"]:
            html += f"""
                <div style="flex: 1; background: rgba(231, 76, 60, 0.1); padding: 10px; border-radius: 8px; border: 1px solid #E74C3C;">
                    <strong style="color: #E74C3C;">🎯 Weaknesses:</strong>
                    <span style="color: #ccc;"> {', '.join(result['weaknesses'])}</span>
                </div>
            """
        html += "</div>"
        
    html += "</div>"
    return html

# ---

def generate_full_report_html(doc_result: DocumentResult) -> str:
    """
    Generates a complete, downloadable HTML report including document summary and 
    all detailed sentence analyses.
    """
    # Helper to find the level for a score based on SCORE_LEVELS dictionary
    def get_level_data(score: float, score_levels: Dict[str, Any]) -> Dict[str, Any]:
        for level_data in score_levels.values():
            if score >= level_data["min"]:
                return level_data
        return score_levels["needs_work"]
    
    # Find strongest/weakest area for the summary metrics
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
#               MOCK DATA (ASSUMED GLOBAL/IMPORTED)
# ============================================================

# NOTE: These dictionaries must be defined for the application to run.
# They are included here as mock data based on the previous context.

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

# --- Mock Analysis Class and Functions (Stubs for App Context) ---
# NOTE: In a real application, these would be imported from the previous sections.
class CriticalThinkingAnalyzer:
    def __init__(self):
        self.standards = PAUL_STANDARDS
        self.score_levels = SCORE_LEVELS
        self.standard_keys = list(PAUL_STANDARDS.keys())

    # Stubs for the complex analysis logic
    def analyze_document(self, sentences: List[str], doc_name: str, doc_id: str) -> Dict[str, Any]:
        # Simple mock logic for demonstration purposes
        if not sentences:
            return {"total_sentences": 0}
            
        # Mock scores based on the input text structure for the demo value
        mock_score = 0.45 
        if "obviously flawed" in sentences[0].lower() or "everyone knows" in sentences[0].lower():
             mock_score = 0.35
        
        standard_averages = {k: mock_score for k in self.standard_keys}
        overall_avg = sum(standard_averages.values()) / len(standard_averages)
        overall_level = SCORE_LEVELS["needs_work"] if overall_avg < 0.5 else SCORE_LEVELS["adequate"]
        
        sentence_results = []
        for i, sentence in enumerate(sentences):
            sentence_score = overall_avg + np.random.uniform(-0.1, 0.1) # Simulate slight variation
            level_data = self._get_score_level(sentence_score)
            
            # Simplified mock for sentence analysis
            standards_analysis = {}
            for k in self.standard_keys:
                standards_analysis[k] = {
                    "standard_name": PAUL_STANDARDS[k]["name"],
                    "score": max(0.0, min(1.0, standard_averages[k] + np.random.uniform(-0.1, 0.1))),
                    "level": self._get_score_level(standard_averages[k]),
                    "color": PAUL_STANDARDS[k]["color"],
                    "icon": PAUL_STANDARDS[k]["icon"],
                    "feedback": f"Mock feedback for {PAUL_STANDARDS[k]['name'].lower()}.",
                    "question": PAUL_STANDARDS[k]['question']
                }

            sentence_results.append({
                "index": i + 1,
                "sentence": sentence,
                "word_count": len(sentence.split()),
                "standards": standards_analysis,
                "overall_score": sentence_score,
                "overall_level": level_data,
                "strengths": [PAUL_STANDARDS["clarity"]["name"]],
                "weaknesses": [PAUL_STANDARDS["fairness"]["name"]],
                "recommendations": []
            })
            
        return {
            "document_name": doc_name,
            "document_id": doc_id,
            "total_sentences": len(sentences),
            "sentence_results": sentence_results,
            "standard_averages": standard_averages,
            "overall_average": overall_avg,
            "overall_level": overall_level
        }

    def _get_score_level(self, score: float) -> Dict[str, Any]:
        for level_data in self.score_levels.values():
            if score >= level_data["min"]:
                return level_data
        return self.score_levels["needs_work"]

# Stubs for other required functions
def extract_text_from_file(file_source: Union[io.BytesIO, Any], file_type: str) -> str:
    # Simplified mock for file extraction
    if file_type == 'pdf':
        try:
            reader = pypdf.PdfReader(file_source)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            return f"ERROR_PDF_EXTRACTION: {e}"
    elif file_type == 'docx':
        try:
            document = docx.Document(file_source)
            return "\n".join(p.text for p in document.paragraphs if p.text)
        except Exception as e:
            return f"ERROR_DOCX_EXTRACTION: {e}"
    elif file_type == 'txt':
        try:
            return file_source.read().decode('utf-8')
        except Exception as e:
            return f"ERROR_TXT_EXTRACTION: {e}"
    else:
        return f"ERROR_UNSUPPORTED_TYPE: {file_type}"

def preprocess_text(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

# Stubs for Visualization and Report Generation (from previous sections)
def create_standard_radar(averages, title): 
    categories = [PAUL_STANDARDS[k]["name"] for k in averages.keys()]
    values = [averages[k] for k in categories] 
    r_values = values + [values[0]]
    theta_categories = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r_values, theta=theta_categories, fill='toself', fillcolor='rgba(102, 126, 234, 0.3)', line=dict(color='#667eea', width=2), name='Score'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0], tickfont=dict(size=10)), angularaxis=dict(tickfont=dict(size=11))), title=title, height=450, template="plotly_dark", margin=dict(t=50, b=50, l=70, r=70))
    return fig

def create_standards_bar_chart(averages, title): 
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

def create_progress_chart(sentence_results, title):
    if not sentence_results: return go.Figure()
    indices = [r["index"] for r in sentence_results]
    overall_scores = [r["overall_score"] for r in sentence_results]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=overall_scores, mode='lines+markers', name='Overall Score', line=dict(color='#667eea', width=3), marker=dict(size=5)))
    x_np, y_np = np.array(indices), np.array(overall_scores)
    z = np.polyfit(x_np, y_np, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=x_np, y=p(x_np), mode='lines', name=f'Trend Line (Slope: {z[0]:.3f})', line=dict(color='#E74C3C', width=2, dash='dash')))
    fig.update_layout(title=title, xaxis_title="Sentence Number", yaxis_title="Score", yaxis=dict(range=[0, 1.05]), height=350, template="plotly_dark", hovermode="x unified")
    return fig

def create_sentence_heatmap(sentence_results, title):
    if not sentence_results: return go.Figure()
    standards_keys = list(PAUL_STANDARDS.keys())
    matrix = [[result["standards"][s]["score"] for s in standards_keys] for result in sentence_results[:30]]
    labels = [f"S{result['index']}" for result in sentence_results[:30]]
    x_labels = [PAUL_STANDARDS[s]["name"] for s in standards_keys]
    fig = px.imshow(matrix, x=x_labels, y=labels, color_continuous_scale="RdYlGn", aspect="auto", title=title)
    fig.update_layout(height=max(400, len(labels) * 25), template="plotly_dark", xaxis=dict(tickangle=45), coloraxis_colorbar=dict(title="Score"))
    return fig

def generate_sentence_report_html(result):
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

def generate_full_report_html(doc_result):
    def get_level_data(score, score_levels):
        for level_data in score_levels.values():
            if score >= level_data["min"]:
                return level_data
        return score_levels["needs_work"]
    
    strongest_standard = max(doc_result['standard_averages'], key=doc_result['standard_averages'].get)
    weakest_standard = min(doc_result['standard_averages'], key=doc_result['standard_averages'].get)

    html = f"""<!DOCTYPE html>
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
            </tr>"""
    
    for key, score in doc_result['standard_averages'].items():
        std = PAUL_STANDARDS[key]
        level_data = get_level_data(score, SCORE_LEVELS)
        html += f"""
            <tr>
                <td style="color: {std['color']}; font-weight: bold;">{std['icon']} {std['name']}</td>
                <td>{score:.0%}</td>
                <td><span class="score-badge" style="background: {level_data['color']};">{level_data['label']}</span></td>
                <td style="color: #888; font-style: italic;">{std['question']}</td>
            </tr>"""
    
    html += """
        </table>
        
        <h2>📝 Detailed Sentence Analysis</h2>
        <p style="color: #aaa;">Review the breakdown for each sentence below.</p>
    """
    
    for result in doc_result['sentence_results']:
        html += generate_sentence_report_html(result)
    
    html += """
    </body>
    </html>
    """
    return html

# ============================================================
#                     MAIN STREAMLIT APP
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
                
                # Reset file pointer for reading
                uploaded_file.seek(0)
                text_result = extract_text_from_file(uploaded_file, file_type)
                
                if text_result.startswith("ERROR_"):
                    st.error(f"File extraction failed: {text_result}")
                else:
                    user_text = text_result
        else:
            default_text = "The current economic policy is obviously flawed. It should be changed immediately because everyone knows a different system will clearly yield better results, but all the politicians are too corrupt to understand the simple solution."
            user_text = st.text_area("Paste Text for Analysis", height=300, value=default_text)
            
        # Add a section to display the raw text for verification
        st.subheader("Raw Text Preview")
        st.code(user_text[:500] + ('...' if len(user_text) > 500 else ''), language='text')

    # --- Main Content Area ---
    if user_text:
        # Preprocess and Analyze
        try:
            sentences = preprocess_text(user_text)
            if not sentences:
                st.warning("The extracted text is too short or could not be properly segmented into sentences (min 10 characters).")
                st.stop()

            # Initialize Analyzer and Run Analysis
            analyzer = CriticalThinkingAnalyzer()
            doc_id = str(hash(user_text)) # Simple deterministic ID
            doc_result = analyzer.analyze_document(sentences, doc_name, doc_id)

            st.success(f"Analysis Complete: {doc_result['total_sentences']} sentences processed from **{doc_name}**.")
            
            # --- 1. Overall Metrics ---
            st.header("📊 Document Overview")
            col1, col2, col3, col4 = st.columns(4)

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

            st.markdown("---")

            # --- 2. Visualization Charts ---
            st.header("📈 Visualization of Standards")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Radar Chart: Excellent for multi-dimensional comparison
                st.plotly_chart(create_standard_radar(doc_result['standard_averages'], "Standards Radar Chart"), use_container_width=True)
                


            with chart_col2:
                # Bar Chart: Best for ranked list/comparison
                st.plotly_chart(create_standards_bar_chart(doc_result['standard_averages'], "Average Score by Standard"), use_container_width=True)

            # Progress Chart: Time series analysis
            st.plotly_chart(create_progress_chart(doc_result['sentence_results'], "Critical Thinking Score Progression Through Document"), use_container_width=True)
            
            if len(doc_result['sentence_results']) > 1:
                # Heatmap: Shows score distribution across many sentences/standards
                st.plotly_chart(create_sentence_heatmap(doc_result['sentence_results'], "Sentence-by-Sentence Score Heatmap (First 30)"), use_container_width=True)

            st.markdown("---")

            # --- 3. Detailed Sentence Breakdown ---
            st.header("📝 Detailed Sentence Breakdown")
            
            # Selector for Sentence
            sentence_options = {r['index']: r['sentence'][:70] + '...' for r in doc_result['sentence_results']}
            selected_index = st.selectbox("Select a Sentence to View Details", options=list(sentence_options.keys()), format_func=lambda x: f"Sentence {x}: {sentence_options[x]}")

            selected_result = next((r for r in doc_result['sentence_results'] if r['index'] == selected_index), None)

            if selected_result:
                st.markdown(generate_sentence_report_html(selected_result), unsafe_allow_html=True)

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
            st.error(f"An unexpected error occurred during analysis: {e}")
            st.exception(e)

    else:
        st.info("Paste or upload a document in the sidebar to begin the Critical Thinking Analysis.")

if __name__ == '__main__':
    main()
