import os
import openai
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from ddgs import DDGS
import uvicorn

# API key from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

openai.api_key = OPENROUTER_API_KEY
ddgs = DDGS()

app = FastAPI()

HTML_TEMPLATE = """
<html>
    <head><title>ResearchBot</title></head>
    <body>
        <h2>ResearchBot WebUI</h2>
        <form action="/" method="post">
            <textarea name="prompt" rows="4" cols="50" placeholder="Enter your research question"></textarea><br>
            <input type="submit" value="Submit">
        </form>
        <div>
            {result}
        </div>
    </body>
</html>
"""

# Generate an optimized search query
def generate_query(user_prompt: str) -> str:
    response = openai.Completion.create(
        model="openrouter/openai/gpt-oss-20b:free",
        prompt=f"Generate an optimized search query for this research question:\n{user_prompt}",
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Conduct research via DDGS
def conduct_research(query: str, num_results: int = 5) -> list:
    results = []
    for r in ddgs.text(query, max_results=num_results):
        results.append({
            "title": r.title,
            "snippet": r.snippet,
            "link": r.url
        })
    return results

# Synthesize research into a final answer
def synthesize_research(user_prompt: str, research: list) -> str:
    sources_text = "\n".join([f"{r['title']}: {r['snippet']}" for r in research])
    prompt = f"Using the following research, answer the question:\n{user_prompt}\n\nResearch:\n{sources_text}"
    response = openai.Completion.create(
        model="openrouter/openai/gpt-oss-20b:free",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Routes
@app.get("/", response_class=HTMLResponse)
def read_form():
    return HTML_TEMPLATE.format(result="")

@app.post("/", response_class=HTMLResponse)
def submit_form(prompt: str = Form(...)):
    query = generate_query(prompt)
    research = conduct_research(query)
    answer = synthesize_research(prompt, research)
    return HTML_TEMPLATE.format(
        result=f"<h3>Query:</h3>{query}<br><h3>Answer:</h3>{answer}"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
