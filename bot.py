import os
import openai
from ddgs import DDGS
from config import DEFAULT_NUM_RESULTS

# Read API key from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

openai.api_key = OPENROUTER_API_KEY

class ResearchBot:
    def __init__(self):
        self.ddgs = DDGS()

    def generate_query(self, user_prompt: str) -> str:
        prompt = f"Generate an optimized search query for this research question:\n{user_prompt}"
        response = openai.Completion.create(
            model="openrouter/openai/gpt-oss-20b:free",  # replace with your OpenRouter model ID
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    def conduct_research(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> list:
        results = []
        for r in self.ddgs.text(query, max_results=num_results):
            results.append({
                "title": r.title,
                "snippet": r.snippet,
                "link": r.url
            })
        return results

    def synthesize_research(self, user_prompt: str, research: list) -> str:
        sources_text = "\n".join([f"{r['title']}: {r['snippet']}" for r in research])
        prompt = f"Using the following research, answer the question:\n{user_prompt}\n\nResearch:\n{sources_text}"
        response = openai.Completion.create(
            model="openrouter/openai/gpt-oss-20b:free",  # same OpenRouter model
            prompt=prompt,
            max_tokens=500
        )
        return response.choices[0].text.strip()

    def run(self, user_prompt: str = None, auto: bool = True):
        if auto and not user_prompt:
            auto_prompts = [
                "Latest trends in AI-generated art",
                "Recent research on quantum computing applications",
            ]
            results = []
            for p in auto_prompts:
                query = self.generate_query(p)
                research = self.conduct_research(query)
                answer = self.synthesize_research(p, research)
                results.append({"prompt": p, "query": query, "answer": answer})
            return results
        elif user_prompt:
            query = self.generate_query(user_prompt)
            research = self.conduct_research(query)
            answer = self.synthesize_research(user_prompt, research)
            return {"prompt": user_prompt, "query": query, "answer": answer}
