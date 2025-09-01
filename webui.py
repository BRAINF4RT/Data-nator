from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
from bot import ResearchBot

app = FastAPI()
bot = ResearchBot()

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

@app.get("/", response_class=HTMLResponse)
def read_form():
    return HTML_TEMPLATE.format(result="")

@app.post("/", response_class=HTMLResponse)
def submit_form(prompt: str = Form(...)):
    result = bot.run(user_prompt=prompt, auto=False)
    return HTML_TEMPLATE.format(result=f"<h3>Query:</h3>{result['query']}<br><h3>Answer:</h3>{result['answer']}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
