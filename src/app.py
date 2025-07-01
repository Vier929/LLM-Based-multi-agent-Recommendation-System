"""
Streamlit Demo for Theory-Driven Multi-Agent Recommendation System
Interactive interface for urban AI recommendations
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import os
import sys
import numpy as np
import importlib.util

# Import the recommendation system components with special handling for + in filename
try:
    # Path to the recommendation+system.py file
    module_path = r"C:\Users\luvyf\Desktop\recommendation+system.py"

    # Check if file exists
    if not os.path.exists(module_path):
        st.error(f"⚠️ File not found: {module_path}")
        st.error("Please ensure recommendation+system.py exists at C:\\Users\\luvyf\\Desktop\\")
        st.stop()

    # Use importlib to import file with special characters
    spec = importlib.util.spec_from_file_location("recommendation_system", module_path)
    if spec is None:
        st.error("⚠️ Failed to load module specification")
        st.stop()

    recommendation_system = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        st.error("⚠️ Module loader is None")
        st.stop()

    spec.loader.exec_module(recommendation_system)

    # Import required classes from the module
    UrbanAIRecommendationSystem = recommendation_system.UrbanAIRecommendationSystem
    UrbanScenario = recommendation_system.UrbanScenario
    UrbanTheory = recommendation_system.UrbanTheory
    Algorithm = recommendation_system.Algorithm
    DataSource = recommendation_system.DataSource
    Recommendation = recommendation_system.Recommendation
    display_recommendation = recommendation_system.display_recommendation

except Exception as e:
    st.error(f"⚠️ Could not import the recommendation system. Error: {str(e)}")
    st.error(f"Error type: {type(e).__name__}")
    st.error(f"Current working directory: {os.getcwd()}")
    import traceback

    st.error(f"Traceback: {traceback.format_exc()}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Urban AI Recommendation System",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 4.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.6rem;
        margin-bottom: 3rem;
        font-weight: 500;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1e3c72;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .recommendation-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
    st.session_state.recommendation = None
    st.session_state.processing_steps = []
    st.session_state.current_challenge = ""
    st.session_state.selected_case = ""
    st.session_state.case_studies = {}


# Helper functions
def initialize_system():
    """Initialize the recommendation system"""
    with st.spinner("🔧 Initializing Urban AI Recommendation System..."):
        try:
            system = UrbanAIRecommendationSystem()
            st.session_state.system = system
            return True
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            return False


def create_agent_status_card(agent_name, status, description="", progress=0):
    """Create a status card for an agent"""
    status_icons = {
        "pending": "⏳",
        "processing": "🔄",
        "completed": "✅",
        "error": "❌"
    }

    icon = status_icons.get(status, "❓")

    col1, col2 = st.columns([1, 9])
    with col1:
        st.markdown(f"<h1>{icon}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{agent_name}**")
        if description:
            st.caption(description)
        if status == "processing" and progress > 0:
            st.progress(progress)


async def process_challenge_async(system, challenge):
    """Process urban challenge asynchronously with status updates"""
    steps = []

    # Stage 1: Scenario Analysis
    st.session_state.processing_steps.append(("Scenario Analyzer", "processing", "Analyzing urban challenge...", 20))
    scenario = await system.scenario_analyzer.process(challenge)
    steps.append({
        "agent": "Scenario Analyzer",
        "result": f"Domain: {scenario.domain}, Complexity: {scenario.complexity_score:.2f}",
        "data": scenario
    })
    st.session_state.processing_steps[-1] = ("Scenario Analyzer", "completed", f"Domain: {scenario.domain}", 100)

    # Stage 2: Theory Retrieval
    st.session_state.processing_steps.append(("Theory Retriever", "processing", "Retrieving relevant theories...", 40))
    theories = await system.theory_retriever.process(scenario)
    steps.append({
        "agent": "Theory Retriever",
        "result": f"Retrieved {len(theories)} theories",
        "data": theories
    })
    st.session_state.processing_steps[-1] = (
    "Theory Retriever", "completed", f"Found {len(theories)} relevant theories", 100)

    # Stage 3: Algorithm Matching
    st.session_state.processing_steps.append(
        ("Algorithm Matcher", "processing", "Matching algorithms to theories...", 60))
    algorithms = await system.algorithm_matcher.process(theories, scenario)
    steps.append({
        "agent": "Algorithm Matcher",
        "result": f"Matched {len(algorithms)} algorithms",
        "data": algorithms
    })
    st.session_state.processing_steps[-1] = (
    "Algorithm Matcher", "completed", f"Matched {len(algorithms)} algorithms", 100)

    # Stage 4: Data Source Selection
    st.session_state.processing_steps.append(("Data Source Selector", "processing", "Selecting data sources...", 80))
    data_sources = await system.data_selector.process(algorithms, scenario)
    steps.append({
        "agent": "Data Source Selector",
        "result": f"Selected {len(data_sources)} data sources",
        "data": data_sources
    })
    st.session_state.processing_steps[-1] = (
    "Data Source Selector", "completed", f"Selected {len(data_sources)} data sources", 100)

    # Stage 5: Integration Validation
    st.session_state.processing_steps.append(
        ("Integration Validator", "processing", "Validating recommendations...", 90))
    recommendation = await system.validator.process(theories, algorithms, data_sources, scenario)
    steps.append({
        "agent": "Integration Validator",
        "result": f"Confidence: {recommendation.confidence_score:.2f}",
        "data": recommendation
    })
    st.session_state.processing_steps[-1] = ("Integration Validator", "completed", f"Validation complete", 100)

    return recommendation, steps


def visualize_theory_distribution(theories):
    """Create a pie chart of theory categories"""
    categories = [t.category for t in theories]
    df = pd.DataFrame({"Category": categories})
    fig = px.pie(df, names="Category", title="Theory Distribution by Category",
                 color_discrete_map={"Safety": "#FF6B6B", "Design": "#4ECDC4",
                                     "Spatial": "#45B7D1", "Perception": "#FFA07A"})
    return fig


def visualize_algorithm_capabilities(algorithms):
    """Create a radar chart of algorithm capabilities"""
    # Aggregate capabilities across all algorithms
    all_capabilities = {}
    for algo in algorithms:
        for cap, score in algo.capabilities.items():
            if cap not in all_capabilities:
                all_capabilities[cap] = []
            all_capabilities[cap].append(score)

    # Calculate average scores
    avg_capabilities = {cap: sum(scores) / len(scores) for cap, scores in all_capabilities.items()}

    # Create radar chart
    categories = list(avg_capabilities.keys())
    values = list(avg_capabilities.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Average Capability Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Algorithm Capabilities Overview"
    )

    return fig


def visualize_data_quality(data_sources):
    """Create a heatmap of data source quality scores"""
    # Prepare data for heatmap
    source_names = [ds.name for ds in data_sources]
    quality_dimensions = list(data_sources[0].quality_scores.keys()) if data_sources else []

    # Create matrix
    quality_matrix = []
    for ds in data_sources:
        scores = [ds.quality_scores.get(dim, 0) for dim in quality_dimensions]
        quality_matrix.append(scores)

    fig = go.Figure(data=go.Heatmap(
        z=quality_matrix,
        x=quality_dimensions,
        y=source_names,
        colorscale='Viridis',
        text=[[f"{score:.2f}" for score in row] for row in quality_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))

    fig.update_layout(
        title="Data Source Quality Assessment",
        xaxis_title="Quality Dimensions",
        yaxis_title="Data Sources",
        height=400
    )

    return fig


def visualize_case_transformation(case_data):
    """Create a visual representation of the transformation process"""
    fig = go.Figure()

    # Add traces for original vs transformed approach
    categories = ['Problem Focus', 'Theory Integration', 'Algorithm Sophistication',
                  'Data Comprehensiveness', 'Outcome Impact']

    # Original approach scores (lower)
    original_scores = [3, 1, 4, 3, 4]  # Simulated scores

    # Theory-driven scores (higher)
    theory_driven_scores = [9, 10, 9, 9, 10]  # Simulated scores

    fig.add_trace(go.Scatterpolar(
        r=original_scores,
        theta=categories,
        fill='toself',
        name='Original Approach',
        line_color='#FF6B6B'
    ))

    fig.add_trace(go.Scatterpolar(
        r=theory_driven_scores,
        theta=categories,
        fill='toself',
        name='Theory-Driven Approach',
        line_color='#45B7D1'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title="Transformation Impact Analysis",
        showlegend=True,
        height=400
    )

    return fig


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header" style="font-size: 4.5rem !important;">🏙️ Urban AI Recommendation System</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header" style="font-size: 1.6rem !important;">Theory-Driven Multi-Agent System for Urban Challenge Solutions</p>',
        unsafe_allow_html=True)

    # Initialize selected_case and selected_example
    selected_case = ""
    selected_example = ""

    # Sidebar
    with st.sidebar:
        st.header("📋 System Overview")
        st.info("""
        This system bridges the theory-practice gap in urban AI applications through:

        • **5 Specialized Agents** working collaboratively
        • **Theory-First Approach** grounding AI in urban planning principles
        • **Comprehensive Validation** ensuring practical feasibility
        """)

        st.header("🔧 Configuration")
        data_path = st.text_input("Data Directory", value=r"C:\Users\luvyf\Desktop")

        if st.button("🚀 Initialize System"):
            if initialize_system():
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system")

        # Example challenges and case studies
        st.header("💡 Examples & Cases")

        # Tab selection for examples vs case studies
        example_type = st.radio("Choose type:", ["Quick Examples", "Research Case Studies"])

        if example_type == "Quick Examples":
            examples = [
                "Reduce crime rates in downtown neighborhoods by improving street lighting and surveillance while maintaining resident privacy",
                "Optimize public transportation routes to reduce traffic congestion during peak hours in the city center",
                "Design a sustainable mixed-use development that promotes walkability and community interaction",
                "Implement smart parking solutions to reduce urban congestion and improve air quality",
                "Create safer pedestrian zones in high-traffic areas using AI-powered traffic management"
            ]
            selected_example = st.selectbox("Select an example:", [""] + examples)
            selected_case = ""  # Clear case selection
        else:
            case_studies = {
                "Case A: Problem-Driven (Food Waste Crisis)": {
                    "challenge": "A major US city discards 10,000 tons of edible food annually while significant populations experience food insecurity. We need to optimize food recovery and distribution to address both waste reduction and food access equity.",
                    "original_approach": "Continuous approximation methods for vehicle routing optimization focusing on minimizing transportation costs",
                    "complexity": 0.82,
                    "transformation": {
                        "theories": ["Urban Metabolism Theory", "Environmental Justice Theory"],
                        "algorithms": ["MOEA/D", "Graph Neural Networks", "Temporal LSTM"],
                        "data_sources": ["Food Bank Networks Database", "Social Vulnerability Index",
                                         "Real-time Food Supply APIs", "Transportation Networks"],
                        "data_focus": "From routes to multi-domain: social vulnerability, food supply, environmental impact",
                        "outcome": "Integrated food security framework balancing efficiency and equity"
                    }
                },
                "Case B: Method-Driven (Urban Heat Island)": {
                    "challenge": "Traditional 3D morphological indices achieve only R²<0.8 in Urban Heat Island modeling. We need to develop enhanced stereoscopic urban morphology metrics with improved prediction accuracy while ensuring practical planning applications.",
                    "original_approach": "New stereoscopic metrics with XGBoost achieving R²=0.95, focused on technical performance",
                    "complexity": 0.76,
                    "transformation": {
                        "theories": ["Urban Climate Theory", "Compact City Theory", "Sustainable Urban Design Theory"],
                        "algorithms": ["Physics-Informed Neural Networks", "Spatial-GCN", "XGBoost", "SHAP"],
                        "data_sources": ["LiDAR Point Clouds", "Thermal Satellite Imagery", "Building Energy Database",
                                         "Demographic Census Data"],
                        "data_focus": "From morphology to comprehensive: LiDAR, thermal, social vulnerability, infrastructure",
                        "outcome": "Vulnerability-aware prediction tool supporting planning decisions"
                    }
                },
                "Case C: Technology-Driven (GRU-CNN for Resilience)": {
                    "challenge": "We have developed a GRU-CNN architecture and need to find meaningful urban applications that leverage its real-time processing and predictive modeling capabilities for disaster resilience.",
                    "original_approach": "GRU-CNN technology demonstration without specific disaster events or theoretical frameworks",
                    "complexity": 0.88,
                    "transformation": {
                        "theories": ["Urban Resilience Theory", "Disaster Risk Reduction Framework",
                                     "Socio-Ecological Systems Theory"],
                        "algorithms": ["GRU-CNN", "Agent-Based Modeling", "Network Analysis", "Reinforcement Learning"],
                        "data_sources": ["IoT Sensor Networks", "Emergency Response Systems", "Social Media Streams",
                                         "Infrastructure Networks GIS"],
                        "data_focus": "From technology demo to comprehensive: hazard sensors, infrastructure networks, social patterns",
                        "outcome": "Community-centered resilience platform with participatory interfaces"
                    }
                }
            }

            selected_case = st.selectbox("Select a case study:", [""] + list(case_studies.keys()))

            if selected_case and selected_case in case_studies:
                case = case_studies[selected_case]
                selected_example = case["challenge"]
                st.session_state.selected_case = selected_case
                st.session_state.case_studies = case_studies

            if selected_case and selected_case in case_studies:
                case = case_studies[selected_case]
                selected_example = case["challenge"]

                # Show case study details
                with st.expander("📋 Case Study Details", expanded=True):
                    st.markdown(f"**Original Approach:** {case['original_approach']}")
                    st.markdown(f"**Complexity Score:** {case['complexity']}")

                    st.markdown("**🔄 Expected Transformation:**")
                    trans = case['transformation']
                    st.markdown(f"- **Theories:** {', '.join(trans['theories'])}")
                    st.markdown(f"- **Algorithms:** {', '.join(trans['algorithms'])}")
                    st.markdown(f"- **Data Sources:** {', '.join(trans['data_sources'])}")
                    st.markdown(f"- **Data Evolution:** {trans['data_focus']}")
                    st.markdown(f"- **Outcome:** {trans['outcome']}")

                    st.info("💡 Run the analysis to see how our system transforms this challenge!")
            else:
                selected_example = ""

        # Clear selection button
        if selected_example or (example_type == "Research Case Studies" and selected_case):
            if st.button("🗑️ Clear Selection", use_container_width=True):
                st.session_state.selected_case = ""
                st.rerun()

    # Main content area
    if st.session_state.system is None:
        st.warning("⚠️ Please initialize the system using the sidebar button.")
        return

    # Input section
    col1, col2 = st.columns([4, 1])
    with col1:
        challenge = st.text_area(
            "🎯 Describe your urban challenge:",
            value=selected_example if selected_example else st.session_state.current_challenge,
            height=100,
            placeholder="Example: We need to reduce crime rates in downtown neighborhoods..."
        )

    with col2:
        st.write("")  # Spacing
        st.write("")
        analyze_button = st.button("🔍 Analyze Challenge", type="primary", use_container_width=True)

    # Processing section
    if analyze_button and challenge:
        st.session_state.current_challenge = challenge
        st.session_state.processing_steps = []

        # Create placeholders for dynamic updates
        status_container = st.container()

        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            with st.spinner("Processing urban challenge..."):
                recommendation, steps = loop.run_until_complete(
                    process_challenge_async(st.session_state.system, challenge)
                )
                st.session_state.recommendation = recommendation

        except Exception as e:
            st.error(f"Error processing challenge: {str(e)}")
            return
        finally:
            loop.close()

        # Display processing steps
        with status_container:
            st.header("🤖 Agent Processing Status")
            for agent_name, status, desc, progress in st.session_state.processing_steps:
                create_agent_status_card(agent_name, status, desc, progress)

        st.success("✅ Analysis complete! See recommendations below.")

    # Results section
    if st.session_state.recommendation:
        st.markdown('<div class="recommendation-section">', unsafe_allow_html=True)
        st.header("📊 Recommendation Results")

        # Confidence metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Score", f"{st.session_state.recommendation.confidence_score:.2%}")
        with col2:
            st.metric("Theories Selected", len(st.session_state.recommendation.theories))
        with col3:
            human_val = "Yes" if st.session_state.recommendation.validation_results[
                'requires_human_validation'] else "No"
            st.metric("Human Validation Required", human_val)

        # Show case study comparison if applicable
        if st.session_state.selected_case and st.session_state.selected_case in st.session_state.case_studies:
            st.markdown("---")
            st.subheader("🔄 Case Study Transformation Analysis")

            case = st.session_state.case_studies[st.session_state.selected_case]

            # Transformation visualization
            transformation_fig = visualize_case_transformation(case)
            st.plotly_chart(transformation_fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 📌 Original Approach")
                st.info(case['original_approach'])

                st.markdown("##### ❌ Limitations")
                if "Case A" in st.session_state.selected_case:
                    st.write("• Focus only on cost minimization")
                    st.write("• Ignores food insecurity patterns")
                    st.write("• No equity considerations")
                elif "Case B" in st.session_state.selected_case:
                    st.write("• Pure technical performance focus")
                    st.write("• No planning integration")
                    st.write("• Limited actionability")
                else:  # Case C
                    st.write("• Technology without application")
                    st.write("• No theoretical grounding")
                    st.write("• Missing community context")

            with col2:
                st.markdown("##### 🎯 Theory-Driven Enhancement")

                # Show expected vs actual theories
                expected_theories = case['transformation']['theories']
                actual_theories = [t.name for t in st.session_state.recommendation.theories]

                st.write("**Expected Theories:**")
                for theory in expected_theories:
                    if any(theory.lower() in t.lower() for t in actual_theories):
                        st.write(f"✅ {theory}")
                    else:
                        st.write(f"⚪ {theory}")

                # Show expected data sources
                st.write("\n**Expected Data Sources:**")
                for ds in case['transformation']['data_sources']:
                    st.write(f"📊 {ds}")

                st.write("\n**Key Improvements:**")
                st.success(case['transformation']['outcome'])

            # Additional metrics comparison
            st.markdown("---")
            st.markdown("##### 📊 Transformation Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric("Complexity", f"{case['complexity']:.2f}", "+0.3")
            with metric_col2:
                st.metric("Theory Integration", "High", "↑")
            with metric_col3:
                st.metric("Data Sources", f"{len(case['transformation']['data_sources'])}",
                          f"+{len(case['transformation']['data_sources'])}")
            with metric_col4:
                st.metric("Algorithm Types", f"{len(case['transformation']['algorithms'])}", "↑")

            # Lessons learned section
            st.markdown("---")
            st.markdown("##### 💡 Key Lessons from Transformation")

            lessons_col1, lessons_col2 = st.columns(2)

            with lessons_col1:
                st.markdown("**🎯 Strategic Insights:**")
                if "Case A" in st.session_state.selected_case:
                    st.write("• Theory grounds technical solutions in social context")
                    st.write("• Multi-objective optimization enables equity integration")
                    st.write("• Comprehensive data reveals hidden vulnerabilities")
                elif "Case B" in st.session_state.selected_case:
                    st.write("• Physical principles enhance ML model reliability")
                    st.write("• Interpretability enables planning applications")
                    st.write("• Social data transforms accuracy into actionability")
                else:  # Case C
                    st.write("• Theory provides purpose for advanced technology")
                    st.write("• Community needs guide technical capabilities")
                    st.write("• Integration creates resilience beyond prediction")

            with lessons_col2:
                st.markdown("**🔧 Implementation Guidance:**")
                if "Case A" in st.session_state.selected_case:
                    st.write("• Start with stakeholder mapping")
                    st.write("• Balance efficiency with equity metrics")
                    st.write("• Design for iterative improvement")
                elif "Case B" in st.session_state.selected_case:
                    st.write("• Embed domain knowledge in models")
                    st.write("• Prioritize actionable outputs")
                    st.write("• Validate with planning practitioners")
                else:  # Case C
                    st.write("• Co-design with communities")
                    st.write("• Build adaptive feedback loops")
                    st.write("• Ensure local knowledge integration")

        # Tabbed results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📚 Theories", "🤖 Algorithms", "📊 Data Sources", "📈 Visualizations", "⚠️ Validation"])

        with tab1:
            st.subheader("Recommended Urban Theories")
            for theory in st.session_state.recommendation.theories:
                with st.expander(f"**{theory.name}** ({theory.year}) - {theory.category}", expanded=True):
                    st.write("**Key Principles:**")
                    for principle in theory.principles:
                        st.write(f"• {principle}")
                    st.write("**Computational Requirements:**")
                    st.write(", ".join(theory.computational_requirements))

        with tab2:
            st.subheader("Recommended AI/ML Algorithms")
            for algo in st.session_state.recommendation.algorithms:
                with st.expander(f"**{algo.name}** (Group {algo.group})", expanded=True):
                    st.write("**Key Capabilities:**")
                    capability_df = pd.DataFrame(
                        [(cap, score) for cap, score in algo.capabilities.items()],
                        columns=["Capability", "Score"]
                    )
                    st.dataframe(capability_df, use_container_width=True)
                    st.write(f"**Computational Cost:** {algo.computational_cost:.2f}")
                    st.write(f"**Data Requirements:** {', '.join(algo.data_requirements)}")

        with tab3:
            st.subheader("Recommended Data Sources")
            for source in st.session_state.recommendation.data_sources:
                with st.expander(f"**{source.name}** ({source.type})", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accessibility", f"{source.accessibility:.1%}")
                    with col2:
                        st.metric("Update Frequency", source.update_frequency)

                    st.write("**Quality Scores:**")
                    quality_df = pd.DataFrame(
                        [(dim, score) for dim, score in source.quality_scores.items()],
                        columns=["Dimension", "Score"]
                    )
                    st.bar_chart(quality_df.set_index("Dimension"))

        with tab4:
            st.subheader("System Analysis Visualizations")

            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.recommendation.theories:
                    fig1 = visualize_theory_distribution(st.session_state.recommendation.theories)
                    st.plotly_chart(fig1, use_container_width=True)

            with col2:
                if st.session_state.recommendation.algorithms:
                    fig2 = visualize_algorithm_capabilities(st.session_state.recommendation.algorithms)
                    st.plotly_chart(fig2, use_container_width=True)

            if st.session_state.recommendation.data_sources:
                fig3 = visualize_data_quality(st.session_state.recommendation.data_sources)
                st.plotly_chart(fig3, use_container_width=True)

        with tab5:
            st.subheader("Integration Validation Results")

            val_results = st.session_state.recommendation.validation_results

            # Robustness score
            st.metric("Robustness Score", f"{val_results['robustness']:.2%}")

            # Compatibility issues
            if val_results['compatibility_issues']:
                st.warning("⚠️ Compatibility Issues Detected:")
                for issue in val_results['compatibility_issues']:
                    st.write(f"• {issue}")
            else:
                st.success("✅ No compatibility issues detected")

            # Human validation recommendation
            if val_results['requires_human_validation']:
                st.info("""
                🧑‍💼 **Human Validation Recommended**

                Based on the confidence score and compatibility analysis, we recommend 
                human expert review before implementation. Key areas to review:
                - Theory-algorithm alignment
                - Data source sufficiency
                - Practical implementation constraints
                """)

        st.markdown('</div>', unsafe_allow_html=True)

        # Export functionality
        st.header("📥 Export Recommendation")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Export as JSON"):
                # Create exportable format
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "challenge": st.session_state.current_challenge,
                    "confidence_score": st.session_state.recommendation.confidence_score,
                    "theories": [{"name": t.name, "category": t.category, "year": t.year}
                                 for t in st.session_state.recommendation.theories],
                    "algorithms": [{"name": a.name, "group": a.group, "capabilities": a.capabilities}
                                   for a in st.session_state.recommendation.algorithms],
                    "data_sources": [{"name": d.name, "type": d.type, "accessibility": d.accessibility}
                                     for d in st.session_state.recommendation.data_sources],
                    "validation": st.session_state.recommendation.validation_results
                }

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"urban_ai_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("📊 Generate Report"):
                st.info("Report generation feature coming soon!")

        with col3:
            if st.button("🔄 New Analysis"):
                st.session_state.recommendation = None
                st.session_state.current_challenge = ""
                st.session_state.selected_case = ""
                st.rerun()


if __name__ == "__main__":
    main()