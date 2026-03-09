# Summarization Agent

AI-powered text summarization using OpenRouter API with Pydantic validation.

## Setup

1. Install dependencies:
```bash
pip install openai python-dotenv pydantic
```

2. Create `.env` file:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

```python
from summerizer import SummarizationAgent

agent = SummarizationAgent(api_key="your_key", model="openai/gpt-oss-120b:free")
result = agent.summarize("Your long text here")

print(result.summary)
print(result.action_items)
print(result.key_decisions)
print(result.key_details)
```

## Output Schema

```python
{
    "summary": str,
    "action_items": list[str],
    "key_decisions": list[str],
    "key_details": list[str]
}
```
