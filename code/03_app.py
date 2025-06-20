# import streamlit as st
# import os
# from PIL import Image

# # Mapping business questions to their plot files and descriptions
# BUSINESS_QUESTIONS = {
#     "Q1 - Drivers of Happiness": {
#         "desc": "Which urban factors have the strongest relationship with happiness?",
#         "plots": [
#             ("q1_correlation_matrix.png", "Correlation Matrix: Happiness vs Urban Factors"),
#             ("q1_green_space_regression.png", "Regression: Happiness vs Green Space"),
#             ("q1_air_quality_regression.png", "Regression: Happiness vs Air Quality"),
#             ("q1_traffic_density_regression.png", "Regression: Happiness vs Traffic Density"),
#             ("q1_noise_level_regression.png", "Regression: Happiness vs Noise Level"),
#         ],
#     },
#     "Q2 - Underperformance": {
#         "desc": "Cities with good infrastructure but low happiness scores - are there outliers?",
#         "plots": [
#             ("q2_infrastructure_vs_happiness.png", "Scatter Plot: Happiness vs Infrastructure Score"),
#             ("q2_residuals.png", "Residual Plot: Identifying underperforming cities"),
#         ],
#     },
#     "Q3 - Temporal Trends": {
#         "desc": "How happiness evolved over time in each city.",
#         "plots": [
#             ("q3_happiness_trends.png", "Time-Series: Happiness Trends Over Time by City"),
#         ],
#     },
#     "Q4 - Green & Clean Impact": {
#         "desc": "Do cities with better green space and air quality have higher happiness?",
#         "plots": [
#             ("q4_environment_boxplot.png", "Boxplot: Environment Score Distribution by Region"),
#             ("q4_environment_score_regression.png", "Regression: Happiness vs Environment Score"),
#         ],
#     },
#     "Q5 - Cost vs Contentment": {
#         "desc": "Is there a tradeoff between cost of living and happiness?",
#         "plots": [
#             ("q5_cost_of_living_regression.png", "Regression: Happiness vs Cost of Living"),
#         ],
#     },
#     "Q6 - Traffic & Noise": {
#         "desc": "Does high traffic density or noise impact happiness negatively?",
#         "plots": [
#             ("q6_traffic_density_regression.png", "Regression: Happiness vs Traffic Density"),
#             ("q6_noise_level_regression.png", "Regression: Happiness vs Noise Level"),
#         ],
#     },
#     "Q7 - Healthcare Importance": {
#         "desc": "How important is healthcare quality to happiness?",
#         "plots": [
#             ("q7_healthcare_score_regression.png", "Regression: Happiness vs Healthcare Quality"),
#         ],
#     },
#     "Q8 - Best & Worst Cities": {
#         "desc": "Which cities consistently rank at the top or bottom for happiness?",
#         "plots": [
#             ("q8_bottom5_cities.png", "Bottom 5 Happiest Cities"),
#             ("q8_top5_cities.png", "Top 5 Happiest Cities"),
#         ],
#     },
# }

# # Path to your figures folder
# FIGURES_PATH = "../figures/analysis"

# def load_image(img_name):
#     img_path = os.path.join(FIGURES_PATH, img_name)
#     if os.path.exists(img_path):
#         return Image.open(img_path)
#     else:
#         return None

# def main():
#     st.set_page_config(page_title="Urban Happiness Dashboard", layout="wide")

#     st.title("🌆 Urban Happiness Index Dashboard")
#     st.markdown(
#         """
#         Explore urban happiness drivers, underperformers, trends, and more.  
#         Use the sidebar to select a business question.
#         """
#     )

#     # Sidebar with selection
#     question = st.sidebar.selectbox("Select Business Question", list(BUSINESS_QUESTIONS.keys()))

#     q_data = BUSINESS_QUESTIONS[question]
#     st.header(question)
#     st.write(q_data["desc"])

#     # Show all plots for selected question
#     for plot_file, caption in q_data["plots"]:
#         img = load_image(plot_file)
#         if img:
#             st.image(img, caption=caption, use_column_width=True)
#         else:
#             st.warning(f"Plot not found: {plot_file}")

#     st.sidebar.markdown("---")
#     st.sidebar.write("Urban Happiness Project by You")

# if __name__ == "__main__":
#     main()


import os
print("Current working directory:", os.getcwd())
