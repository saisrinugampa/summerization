import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class SummarizationAgent:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b:free"):
        """
        Initialize the agent with OpenRouter configuration.
        
        :param api_key: Your OpenRouter API Key.
        :param model: The model ID to use (default: "gpt-oss"). 
                      Note: Ensure 'gpt-oss' is a valid ID in OpenRouter or 
                      replace with a specific ID like 'meta-llama/llama-3-8b-instruct'.
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def summarize(self, text: str, max_tokens: int = 500) -> str:
        """
        Summarizes the provided text using the specified model.

        :param text: The long text to summarize.
        :param max_tokens: The maximum length of the summary.
        :return: The generated summary.
        """
        if not text:
            return "No text provided to summarize."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert summarization agent. "
                            "Your goal is to extract the key points from the provided text "
                            "and present them in a concise, clear, and objective manner. "
                            "Ignore irrelevant details and focus on the core message."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following text:\n\n{text}"
                    }
                ],
                # OpenRouter specific headers can be passed via extra_headers if needed
                extra_headers={
                    "HTTP-Referer": "https://localhost:3000", # Optional: Your site URL
                    "X-Title": "Summarization Agent",         # Optional: Your app name
                },
                max_tokens=max_tokens,
                temperature=0.5, # Lower temperature for more factual summaries
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"An error occurred during summarization: {str(e)}"

# --- Usage Example ---
if __name__ == "__main__":
    # Ideally, load this from an environment variable for security
    # os.environ["OPENROUTER_API_KEY"] = "sk-or-..." 
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("Error: Please set the OPENROUTER_API_KEY environment variable.")
    else:
        # Initialize the agent
        # You can swap "gpt-oss" with specific models like "mistralai/mistral-7b-instruct"
        agent = SummarizationAgent(api_key=api_key, model="openai/gpt-oss-120b:free")

        long_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the 
        natural intelligence displayed by animals including humans. AI research has been defined 
        as the field of study of intelligent agents, which refers to any system that perceives 
        its environment and takes actions that maximize its chance of achieving its goals.
        The term "artificial intelligence" had previously been used to describe machines 
        that mimic and display "human" cognitive skills that are associated with the human 
        mind, such as "learning" and "problem-solving". This definition has since been 
        rejected by major AI researchers who now describe AI in terms of rationality and 
        acting rationally, which does not limit how intelligence can be articulated.
        """

        print("--- Generating Summary ---")
        summary = agent.summarize(long_text)
        print(summary)
