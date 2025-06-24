# Urban Happiness Index Dashboard
# This Streamlit app visualizes urban happiness data, allowing users to explore various business questions
# and their corresponding plots. Users can select different plot types and business questions to analyze urban factors affecting happiness.

# Run this app using the command in the terminal:
# streamlit run C:/Users/julia/OneDrive/Desktop/Projects/Project-City-Happiness-Index/code/03_app.py



# Import necessary libraries
import os
import base64
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

# Mapping business questions to their plot files and descriptions
BUSINESS_QUESTIONS = {
    "Q1 - Drivers of Happiness": {
        "desc": "Which urban factors have the strongest relationship with happiness?",
        "plots": [
            ("business_output/q1_correlation_matrix.png", "Correlation Matrix: Happiness vs Urban Factors"),
            ("regression_output/regression_happiness_score_vs_green_space_area.png", "Regression: Happiness vs Green Space"),
            ("regression_output/regression_happiness_score_vs_air_quality_index.png", "Regression: Happiness vs Air Quality"),
            ("regression_output/regression_happiness_score_vs_traffic_density_numeric.png", "Regression: Happiness vs Traffic Density"),
            ("regression_output/regression_happiness_score_vs_decibel_level.png", "Regression: Happiness vs Decibel Level"),

        ],
        "conclusion": """
    - Environment score (green space area + air quality index), green space show the **strongest positive correlation** with happiness.
    - Traffic density and noise levels (decibel level) have a **moderate to strong negative relationship** with happiness.
    - These factors can guide city planning efforts: improving green areas and pollution control can directly impact citizen happiness.
    """,

    },
    "Q2 - Underperformance": {
        "desc": "Cities with good infrastructure but low happiness scores - are there outliers?",
        "plots": [
            ("business_output/q2_infra_vs_happiness_regression.png", "Regression: Happiness vs Infrastructure Score"),
            ("business_output/q2_residual_overperformers.png", "Residual Plot: Identifying overperforming cities"),
            ("business_output/q2_residual_underperformers.png", "Residual Plot: Identifying underperforming cities"),

        ],
        "conclusion": """
    - Some cities like *CityX* and *CityY* have **strong infrastructure but low happiness**, indicating a mismatch.
    - Potential causes could be non-infrastructure-related: high cost of living, poor air quality, or noise levels.
    - These outliers may need deeper qualitative study to understand why infrastructure alone isn’t enough.
    """,

    },
    "Q3 - Temporal Trends": {
        "desc": "How happiness evolved over time in each city.",
        "plots": [
            ("business_output/q3_happiness_trends_by_city.png", "Time-Series: Happiness Trends Over Time by City"),

        ],
        "conclusion": """
    - Most cities show a **stable or slightly upward trend** in happiness over time, suggesting gradual improvement.
    - A few cities (e.g., *CityZ*) show **sharp dips or fluctuations**, which may correlate with events like policy changes or environmental shifts.
    - Continuous monitoring over time can help evaluate the **impact of interventions** like traffic control or green space development.
    """,

    },
    "Q4 - Green & Clean Impact": {
        "desc": "Do cities with better green space and air quality have higher happiness?",
        "plots": [
            ("business_output/q4_environment_boxplot.png", "Boxplot: Environment Score Distribution by Region"),
            ("business_output/q4_environment_score_regression.png", "Regression: Happiness vs Environment Score"),

        ],
        "conclusion": """
    - Cities with **higher environment scores (air quality + green space)** consistently report **higher happiness scores**.
    - This supports the idea that **urban livability and environmental health are tightly linked** to emotional well-being.
    - Prioritizing clean, green environments could serve as a **cost-effective lever** to improve urban happiness.
    """,

    },
    "Q5 - Cost vs Contentment": {
        "desc": "Is there a tradeoff between cost of living and happiness?",
        "plots": [
            ("regression_output/regression_happiness_score_vs_cost_of_living_index.png", "Regression: Happiness vs Cost of Living"),

        ],
        "conclusion": """
    - There is a **moderate negative correlation** between cost of living and happiness: **higher cost cities tend to have lower happiness**.
    - However, this trend is **not absolute** — a few high-cost cities still maintain good happiness scores, possibly due to high income, services, or safety.
    - Cost-efficiency (value for money) and affordability appear to be more important than just cost alone.
    """,

    },
    "Q6 - Traffic & Noise": {
        "desc": "Does high traffic density or noise impact happiness negatively?",
        "plots": [
            ("regression_output/regression_happiness_score_vs_traffic_density_numeric.png", "Regression: Happiness vs Traffic Density"),
            ("regression_output/regression_happiness_score_vs_decibel_level.png", "Regression: Happiness vs Decibel Level"),

        ],
        "conclusion": """
    - Both **traffic density and noise pollution** show **clear negative impacts on happiness**.
    - Cities with high traffic congestion or decibel levels tend to have lower happiness scores, highlighting quality-of-life concerns.
    - Urban planning strategies like traffic calming, noise regulation, and public transport upgrades can yield significant benefits.
    """,

    },
    "Q7 - Healthcare Importance": {
        "desc": "How important is healthcare quality to happiness?",
        "plots": [
            ("regression_output/regression_happiness_score_vs_healthcare_index.png", "Regression: Happiness vs Healthcare Quality"),

        ],
        "conclusion": """
    - Healthcare quality has a **strong positive relationship** with happiness.
    - Cities with better healthcare systems report **consistently higher happiness scores**, underlining the role of physical well-being in emotional satisfaction.
    - Investment in healthcare infrastructure may serve both practical and emotional outcomes for urban populations.
    """,

    },
    "Q8 - Best & Worst Cities": {
        "desc": "Which cities consistently rank at the top or bottom for happiness?",
        "plots": [
            ("business_output/q8_bottom5_cities.png", "Bottom 5 Happiest Cities"),
            ("business_output/q8_top5_cities.png", "Top 5 Happiest Cities"),
        ],
        "conclusion": """
    - The top 5 cities share common traits: **high environment scores, good healthcare, low noise and traffic, and moderate cost of living**.
    - The bottom 5 cities tend to struggle with **pollution, congestion, or affordability issues**.
    - These rankings can serve as benchmarks for improvement or replication of best practices in other cities.
    """,

    },
}

# Define the folder paths for different plot types
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# Function to load images from the selected folder
def load_image(img_name):
    """Load an image from the figures folder."""
    img_path = os.path.join(BASE_OUTPUT_DIR, img_name)
    # st.write(f"Looking for image at: `{img_path}`")  # DEBUG 
    if os.path.exists(img_path):
        return Image.open(img_path)
    else:
        return None

# Function to display a responsive image with caption
def display_responsive_image(img_path, caption=""):
    if not os.path.exists(img_path):
        st.warning(f"Image not found: {img_path}")
        return

    # Read and encode the image
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Create base64 URI
    img_base64 = f"data:image/png;base64,{encoded_string}"

    # Inject responsive HTML
    html_code = f"""
    <div style="text-align:center; margin-bottom: 2em;">
        <img src="{img_base64}"
             style="max-width: 100%; width: 1000px; height: auto; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.1);">
        <div style="font-size: 0.9rem; color: #666; margin-top: 0.5em;">{caption}</div>
    </div>
    """
    components.html(html_code, height=600)



# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Urban Happiness Dashboard", layout="wide")

    st.title("🌆 Urban Happiness Index Dashboard")
    st.markdown(
        """
        Explore urban happiness drivers, underperformers, trends, and more.  
        Use the sidebar to select a business question.
        """
    )

    # Sidebar with selection
    question = st.sidebar.selectbox("Select Business Question", list(BUSINESS_QUESTIONS.keys()))

    q_data = BUSINESS_QUESTIONS[question]
    st.header(question)
    st.write(q_data["desc"])

    # Show all plots for selected question
    for plot_file, caption in q_data["plots"]:
        img = load_image(plot_file)
        if img:
            # st.image(img, caption=caption, use_container_width=True)
            img_path = os.path.join(BASE_OUTPUT_DIR, plot_file)
            display_responsive_image(img_path, caption)


        else:
            st.warning(f"Plot not found: {plot_file}")

    # Show conclusion after plots
    if "conclusion" in q_data:
        st.subheader("Conclusion")  
        st.write(q_data["conclusion"])

    st.sidebar.markdown("---")
    st.sidebar.write("Urban Happiness Project by Shweta Pandey.")

if __name__ == "__main__":
    main()



