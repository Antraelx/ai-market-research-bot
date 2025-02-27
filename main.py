import requests
import openai
import pandas as pd
import json
import streamlit as st
import matplotlib.pyplot as plt
import os
import time

def get_search_results(query, num_results=5):
    """Fetches Google search results using SerpAPI."""
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "num": num_results,
        "api_key": SERPAPI_KEY
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error("Error fetching search results. Please check your API key and connection.")
        return []
    
    data = response.json()
    results = []
    for item in data.get("organic_results", []):
        results.append({"Title": item.get("title", "No title"), "Link": item.get("link", "No link"), "Snippet": item.get("snippet", "No description available")})
    
    return results

def summarize_data(data):
    """ Uses OpenAI to generate a summary of the data. """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    if not data:
        return "No relevant data found for analysis."
    
    content = "\n".join([f"{item['Title']}: {item['Snippet']} ({item['Link']})" for item in data])
    prompt = f"Perform a competitive analysis based on these results: {content}"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are an AI specializing in competitive analysis."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI summary: {e}"

def save_results(data, summary):
    """ Saves results to a CSV and JSON file. """
    df = pd.DataFrame(data)
    df.to_csv("competitive_analysis.csv", index=False)
    with open("competitive_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"results": data, "summary": summary}, f, indent=4)

def visualize_data(data):
    """ Creates a simple bar chart for competitor analysis. """
    titles = [item["Title"] for item in data]
    values = range(len(data))
    
    fig, ax = plt.subplots()
    ax.barh(titles, values, color='skyblue')
    ax.set_xlabel("Competitors")
    ax.set_ylabel("Ranking")
    ax.set_title("Competitor Analysis Visualization")
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="AI Market Research Bot", layout="wide")
    
    # Custom CSS for modern UI with dark/light mode toggle and animations
    st.markdown("""
        <style>
        .stApp {
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #1e88e5;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 18px;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #1565c0;
            transform: scale(1.05);
        }
        .light-mode {
            background-color: #ffffff;
            color: #000000;
        }
        .dark-mode {
            background-color: #121212;
            color: #ffffff;
            transition: all 0.5s ease-in-out;
        }
        .fade-in {
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .custom-table th, .custom-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .custom-table th {
            background-color: #1e1e1e;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Toggle dark/light mode
    mode = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
    theme_class = "light-mode" if mode == "Light" else "dark-mode"
    
    st.markdown(f'<div class="{theme_class} fade-in">', unsafe_allow_html=True)
    st.title("🔍 AI Market Research Bot")
    st.markdown("## Easily analyze market competition using AI-powered insights.")
    
    with st.container():
        st.markdown("### 🔎 Enter a keyword for analysis")
        query = st.text_input("Keyword:")
        analyze_button = st.button("🚀 Analyze")
    
    if analyze_button and query:
        with st.spinner("🔎 Searching for competitors..."):
            time.sleep(2)
            data = get_search_results(query)
        
        if data:
            with st.spinner("✅ Generating AI report..."):
                time.sleep(2)
                summary = summarize_data(data)
                save_results(data, summary)
            
            st.markdown("## 📊 AI Competitive Analysis Report", unsafe_allow_html=True)
            st.markdown(f"<div class='{theme_class} fade-in'>{summary}</div>", unsafe_allow_html=True)
            
            df = pd.DataFrame(data)
            st.markdown("### 📌 Competitor Data")
            st.table(df)
            
            visualize_data(data)
            
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name="competitive_analysis.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No results found. Try a different search term.")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
