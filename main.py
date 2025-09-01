from bot import ResearchBot

def main():
    bot = ResearchBot()
    
    auto_results = bot.run(auto=True)
    for r in auto_results:
        print(f"Prompt: {r['prompt']}")
        print(f"Query: {r['query']}")
        print(f"Answer:\n{r['answer']}\n{'='*50}\n")

if __name__ == "__main__":
    main()
