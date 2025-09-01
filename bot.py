import time
import os
from openai import OpenAI
from ddgs import DDGS
from config import OPENROUTER_API_KEY, DEFAULT_NUM_RESULTS

# Initialize OpenAI client for OpenRouter (only once)
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Initialize DDGS
ddgs = DDGS()

class ResearchBot:
    def __init__(self):
        pass

    def generate_query(self, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "You are a query optimization assistant. Return only a single concise search query."},
                {"role": "user", "content": f"Generate a DuckDuckGo search query for: {user_prompt}"}
            ],
            max_tokens=50,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    import time

def conduct_research(self, query: str, max_results: int = 50, log_raw: bool = False):
    all_results = []
    seen_urls = set()
    try:
        for r in ddgs.text(query, max_results=max_results):
            link = r.get("href") or r.get("url")
            if link and link not in seen_urls:
                title = r.get("title") or (r.get("text")[:50] if r.get("text") else "No title")
                snippet = r.get("body") or (r.get("text")[:150] if r.get("text") else "No snippet")
                all_results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link
                })
                seen_urls.add(link)
        if log_raw:
            print(f"[RAW SEARCH] Query: '{query}' | Results fetched: {len(all_results)}")
    except Exception as e:
        print(f"Error conducting research: {e}")
    finally:
        time.sleep(1)  # simple rate-limit between queries
    return all_results

    def synthesize_research(self, user_prompt: str, research: list) -> str:
        sources_text = "\n".join([f"{r['title']}: {r['snippet']}" for r in research])
        prompt = f"Using the following research, answer the question:\n{user_prompt}\n\nResearch:\n{sources_text}"

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

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
