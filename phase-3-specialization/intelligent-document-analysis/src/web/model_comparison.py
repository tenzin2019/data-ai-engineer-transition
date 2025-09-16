"""
Model comparison and recommendation tool for the Streamlit app.
"""

import streamlit as st
from utils.model_selector import model_selector


def show_model_comparison():
    """Display model comparison and recommendations."""
    st.subheader("🤖 AI Model Comparison & Recommendations")
    
    # Get model information
    models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    # Create columns for comparison
    col1, col2, col3 = st.columns(3)
    
    for i, model in enumerate(models):
        with [col1, col2, col3][i]:
            info = model_selector.get_model_info(model)
            
            st.markdown(f"### {info['name']}")
            st.markdown(f"**{info['description']}**")
            
            # Cost information
            st.markdown("**Cost per 1K tokens:**")
            st.markdown(f"• Input: ${info['cost_per_1k_tokens']['input']:.4f}")
            st.markdown(f"• Output: ${info['cost_per_1k_tokens']['output']:.4f}")
            
            # Performance metrics
            st.markdown("**Performance:**")
            st.markdown(f"• Reliability: {info['reliability']*100:.0f}%")
            st.markdown(f"• Speed: {info['speed']}")
            
            # Best for
            st.markdown("**Best for:**")
            for use_case in info['best_for']:
                st.markdown(f"• {use_case}")
    
    # Cost calculator
    st.subheader("💰 Cost Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_tokens = st.number_input("Input Tokens", min_value=0, value=1000, step=100)
        output_tokens = st.number_input("Output Tokens", min_value=0, value=250, step=25)
    
    with col2:
        documents_per_day = st.number_input("Documents per Day", min_value=1, value=10, step=1)
        days_per_month = st.number_input("Days per Month", min_value=1, value=22, step=1)
    
    # Calculate costs
    st.markdown("### Cost Analysis")
    
    cost_data = []
    for model in models:
        daily_cost = model_selector.estimate_cost(model, input_tokens, output_tokens) * documents_per_day
        monthly_cost = daily_cost * days_per_month
        
        cost_data.append({
            'model': model_selector.get_model_info(model)['name'],
            'per_document': model_selector.estimate_cost(model, input_tokens, output_tokens),
            'daily': daily_cost,
            'monthly': monthly_cost
        })
    
    # Display cost table
    import pandas as pd
    df = pd.DataFrame(cost_data)
    df.columns = ['Model', 'Per Document', 'Daily Cost', 'Monthly Cost']
    df['Per Document'] = df['Per Document'].apply(lambda x: f"${x:.4f}")
    df['Daily Cost'] = df['Daily Cost'].apply(lambda x: f"${x:.2f}")
    df['Monthly Cost'] = df['Monthly Cost'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(df, use_container_width=True)
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    
    # Find best model for different scenarios
    scenarios = [
        ("High Quality Required", "gpt-4o", "Complex documents, legal, financial, technical"),
        ("Balanced Performance", "gpt-4o-mini", "General business documents, standard analysis"),
        ("Cost Optimized", "gpt-3.5-turbo", "Simple documents, high volume processing")
    ]
    
    for scenario, model, description in scenarios:
        with st.expander(f"**{scenario}** - {model_selector.get_model_info(model)['name']}"):
            st.markdown(f"**Use when:** {description}")
            
            # Show cost for this scenario
            scenario_cost = model_selector.estimate_cost(model, input_tokens, output_tokens)
            st.markdown(f"**Cost per document:** ${scenario_cost:.4f}")
            st.markdown(f"**Monthly cost:** ${scenario_cost * documents_per_day * days_per_month:.2f}")
            
            # Show model details
            info = model_selector.get_model_info(model)
            st.markdown(f"**Reliability:** {info['reliability']*100:.0f}%")
            st.markdown(f"**Speed:** {info['speed']}")


def show_smart_selection_demo():
    """Show how smart model selection works."""
    st.subheader("🧠 Smart Model Selection Demo")
    
    st.markdown("""
    Our system automatically selects the best model based on:
    - **Document type** (legal, financial, technical, etc.)
    - **Document length** (character count)
    - **Complexity score** (technical terms, structure, etc.)
    - **Cost optimization** (budget vs. quality trade-offs)
    """)
    
    # Demo interface
    col1, col2 = st.columns(2)
    
    with col1:
        doc_type = st.selectbox(
            "Document Type",
            ["general", "legal", "financial", "technical", "medical", "business"],
            key="demo_doc_type"
        )
        
        text_length = st.number_input(
            "Document Length (characters)",
            min_value=100,
            max_value=100000,
            value=5000,
            step=100,
            key="demo_length"
        )
    
    with col2:
        complexity = st.slider(
            "Complexity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="demo_complexity"
        )
        
        user_preference = st.selectbox(
            "User Preference (Optional)",
            ["Auto-select", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            key="demo_preference"
        )
    
    # Get recommendation
    if st.button("Get Recommendation", key="demo_button"):
        preference = None if user_preference == "Auto-select" else user_preference
        
        recommendation = model_selector.get_recommendation(doc_type, text_length)
        
        st.success(f"**Recommended Model:** {recommendation['recommended_model']}")
        
        # Show reasoning
        st.info(f"**Reasoning:** {recommendation['reasoning']}")
        
        # Show cost estimate
        st.metric("Estimated Cost", f"${recommendation['estimated_cost']:.4f}")
        st.metric("Estimated Tokens", f"{recommendation['estimated_tokens']:,}")
        
        # Show model details
        model_info = recommendation['model_info']
        st.markdown("**Model Details:**")
        st.markdown(f"• **Reliability:** {model_info['reliability']*100:.0f}%")
        st.markdown(f"• **Speed:** {model_info['speed']}")
        st.markdown(f"• **Best for:** {', '.join(model_info['best_for'])}")
