import os
import time
import requests
from ddgs import DDGS
from openai import OpenAI
from config import OPENROUTER_API_KEY

# Initialize clients
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
ddgs = DDGS()


class ResearchBot:
    def __init__(self, autoprompts_file="auto_prompts.txt", custom_sources_file="custom_sources.txt"):
        self.autoprompts_file = autoprompts_file
        self.custom_sources_file = custom_sources_file

    # -----------------------------
    # Auto-prompts
    # -----------------------------
    def read_autoprompts(self):
        """Read auto prompts from a text file, one prompt per line."""
        if not os.path.exists(self.autoprompts_file):
            print(f"[Warning] Auto-prompts file not found: {self.autoprompts_file}")
            return []
        with open(self.autoprompts_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # -----------------------------
    # Generate optimized query
    # -----------------------------
    def generate_query(self, user_prompt: str) -> str:
        """Generate a concise DuckDuckGo search query using OpenRouter."""
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "You are a query optimization assistant. Return a single concise search query optimized for DuckDuckGo."},
                {"role": "user", "content": f"Generate a DuckDuckGo search query for: {user_prompt}"}
            ],
            max_tokens=50,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    # -----------------------------
    # Load custom URLs
    # -----------------------------
    def load_custom_sources(self):
        """Load URLs from a .txt file and fetch content."""
        results = []
        if not os.path.exists(self.custom_sources_file):
            return results
        with open(self.custom_sources_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        for url in urls:
            try:
                r = requests.get(url, timeout=5)
                r.raise_for_status()
                text = r.text
                results.append({
                    "title": url,
                    "snippet": text,
                    "link": url
                })
            except Exception as e:
                print(f"[Warning] Failed to fetch {url}: {e}")
        return results

    # -----------------------------
    # Conduct research
    # -----------------------------
    def conduct_research(self, query: str, max_results: int = 50, log_raw: bool = False):
        """Combine DuckDuckGo text/news search results with custom URLs."""
        all_results = []
        seen_urls = set()
        # --- Load custom sources first ---
        custom_results = self.load_custom_sources()
        for r in custom_results:
            all_results.append(r)
            seen_urls.add(r["link"])
        try:
            # --- DDGS text results ---
            for r in ddgs.text(query, max_results=max_results):
                link = r.get("href") or r.get("url")
                if link and link not in seen_urls:
                    all_results.append({
                        "title": r.get("title") or (r.get("text")[:50] if r.get("text") else "No title"),
                        "snippet": r.get("body") or (r.get("text")[:150] if r.get("text") else "No snippet"),
                        "link": link
                    })
                    seen_urls.add(link)
            # --- DDGS news results ---
            for r in ddgs.news(query, max_results=max_results):
                link = r.get("href") or r.get("url")
                if link and link not in seen_urls:
                    all_results.append({
                        "title": r.get("title") or (r.get("text")[:50] if r.get("text") else "No title"),
                        "snippet": r.get("body") or (r.get("text")[:150] if r.get("text") else "No snippet"),
                        "link": link
                    })
                    seen_urls.add(link)
            if log_raw:
                print(f"[RAW SEARCH] Query: '{query}' | Total results: {len(all_results)}")
        except Exception as e:
            print(f"[Error] conducting research: {e}")
        finally:
            time.sleep(1)  # rate-limit
        return all_results[:max_results]

    # -----------------------------
    # Synthesize research
    # -----------------------------
    def synthesize_research(self, user_prompt: str, research: list) -> str:
        """Summarize research using OpenRouter + user context."""
        sources_text = "\n".join([f"{r['title']}: {r['snippet']}" for r in research])
        manual_context = self.read_context()
        prompt = f"""
        You are a research assistant.
        User question: {user_prompt}
        --- User Context (manual notes) ---
        {manual_context if manual_context else "No extra context provided."}
        --- Web Research ---
        {sources_text}
        """
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",  # keep same model you used
            messages=[{"role": "system", "content": "You are a research assistant."},
                      {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    # -----------------------------
    # Run bot
    # -----------------------------
    def run(self, user_prompt: str = None, auto: bool = True):
        if auto and not user_prompt:
            auto_prompts = self.read_autoprompts()
            if not auto_prompts:
                print("[Info] No auto-prompts found. Exiting auto mode.")
                return []
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
