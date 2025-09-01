from ddgs import DDGS
from openrouter import OpenRouterLLM
from config import OPENROUTER_API_KEY, DEFAULT_NUM_RESULTS

class ResearchBot:
    def __init__(self):
        # Initialize LLMs
        self.querier_llm = OpenRouterLLM(model="openrouter/openai/gpt-oss-20b:free", api_key=OPENROUTER_API_KEY)
        self.researcher_llm = OpenRouterLLM(model="openrouter/openai/gpt-oss-20b:free", api_key=OPENROUTER_API_KEY)
        # Initialize DDGS
        self.ddgs = DDGS()

    def generate_query(self, user_prompt: str) -> str:
        prompt = f"Generate an optimized search query for the research question:\n{user_prompt}"
        query = self.querier_llm.complete(prompt)
        return query.strip()

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
        prompt = (
            f"Using the following research, answer the question:\n{user_prompt}\n\n"
            f"Research:\n{sources_text}"
        )
        return self.researcher_llm.complete(prompt)

    def run(self, user_prompt: str = None, auto: bool = True):
        """
        If auto=True, runs in fully automated mode with pre-defined queries.
        If user_prompt is given, it runs interactively for that prompt.
        """
        if auto and not user_prompt:
            # Example automated queries (replace with your workflow logic)
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
