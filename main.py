import requests
import openai
import pandas as pd
import json
import streamlit as st
import matplotlib.pyplot as plt
import os

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
        response = openai.ChatCompletion.create(
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
    st.title("🔍 AI Market Research Bot")
    st.markdown("## Easily analyze market competition using AI-powered insights.")
    
    query = st.text_input("### Enter a keyword or phrase for competitive analysis:")
    col1, col2 = st.columns([3,1])
    
    with col2:
        analyze_button = st.button("🚀 Analyze")
    
    if analyze_button and query:
        st.info("🔎 Searching for competitors...")
        data = get_search_results(query)
        
        if data:
            st.success("✅ Generating AI report...")
            summary = summarize_data(data)
            save_results(data, summary)
            
            st.subheader("📊 AI Competitive Analysis Report")
            st.write(summary)
            
            df = pd.DataFrame(data)
            st.write("### 📌 Competitor Data:")
            st.dataframe(df, use_container_width=True)
            
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

if __name__ == "__main__":
    main()
