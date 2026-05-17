import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Movie Analysis Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF4B4B, #FF9B4B);
        -webkit-background-clip: text;
        -webkit-text-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
    .card-profit {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .card-recommend {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Title
    st.markdown('<div class="main-title">🎬 Movie Analysis Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rating Prediction Model | Smart Movie Recommendation</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Use column layout to display two feature cards
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Rating Prediction Model card
        with st.container():
            st.markdown("""
            <div class="card card-profit" style="margin-bottom: 1.5rem;">
                <div class="card-title">💰 Rating Prediction Model</div>
                <div class="card-desc">Predict whether a movie will be profitable based on machine learning models</div>
                <div class="card-desc" style="margin-top: 0.5rem;">← Click the button below to enter</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Enter Rating Prediction Model →", key="profit_btn", use_container_width=True):
                st.session_state.page = "profit"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Movie recommendation card
        with st.container():
            st.markdown("""
            <div class="card card-recommend" style="margin-bottom: 1.5rem;">
                <div class="card-title">🎯 Smart Movie Recommendation</div>
                <div class="card-desc">Enter your favorite movie and get similar movie recommendations</div>
                <div class="card-desc" style="margin-top: 0.5rem;">← Click the button below to enter</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Enter Recommendation System →", key="recommend_btn", use_container_width=True):
                st.session_state.page = "recommend"
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Based on TMDB Movie Dataset | Hybrid Recommendation Engine | XGBoost Profit Prediction Model"
        "</div>",
        unsafe_allow_html=True
    )


# Page routing
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    main()
elif st.session_state.page == "profit":
    with open("success_model.py", "r", encoding="utf-8") as f:
        exec(f.read())
elif st.session_state.page == "recommend":
    with open("recommend_model.py", "r", encoding="utf-8") as f:
        exec(f.read())
