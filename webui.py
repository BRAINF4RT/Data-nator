import os
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
from ddgs import DDGS
from openrouter import OpenRouterLLM

# Read API key from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

# FastAPI app
app = FastAPI()

# Initialize LLMs using the API key
querier_llm = OpenRouterLLM(model="querier-model", api_key=OPENROUTER_API_KEY)
researcher_llm = OpenRouterLLM(model="researcher-model", api_key=OPENROUTER_API_KEY)
ddgs = DDGS()

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

def generate_query(user_prompt: str) -> str:
    prompt = f"Generate an optimized search query for this research question:\n{user_prompt}"
    return querier_llm.complete(prompt).strip()

def conduct_research(query: str, num_results: int = 5) -> list:
    results = []
    for r in ddgs.text(query, max_results=num_results):
        results.append({"title": r.title, "snippet": r.snippet, "link": r.url})
    return results

def synthesize_research(user_prompt: str, research: list) -> str:
    sources_text = "\n".join([f"{r['title']}: {r['snippet']}" for r in research])
    prompt = f"Using the following research, answer the question:\n{user_prompt}\n\nResearch:\n{sources_text}"
    return researcher_llm.complete(prompt)

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
