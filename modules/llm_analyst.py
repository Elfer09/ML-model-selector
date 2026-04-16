"""
llm_analyst.py
Wraps the OpenAI chat API. Injects dataset context into the system prompt
so the LLM can answer questions grounded in the actual data.
"""
import pandas as pd
from openai import OpenAI


def _build_system_prompt(df: pd.DataFrame, profile: dict) -> str:
    """Build a context-rich system prompt from the dataset profile."""
    return f"""You are an expert data analyst. You have been given access to a dataset and your job is to answer the user's questions about it clearly, concisely, and accurately.

DATASET OVERVIEW:
- Shape: {profile['shape'][0]} rows × {profile['shape'][1]} columns
- Numeric columns: {', '.join(profile['numeric_col_names']) or 'None'}
- Categorical columns: {', '.join(profile['categorical_col_names']) or 'None'}
- Missing data: {profile['missing_pct']:.2f}% of all cells

COLUMN DATA TYPES:
{profile['dtypes']}

STATISTICAL SUMMARY:
{profile['describe']}

SAMPLE DATA (first 5 rows):
{profile['sample']}

INSTRUCTIONS:
- Answer based only on the data shown above.
- Be specific: reference column names, numbers, percentages.
- If you spot data quality issues, mention them.
- Format your answers with markdown (bold, bullet points, tables where helpful).
- If a question can't be answered from the data, say so clearly.
- Keep answers concise — lead with the insight, then the explanation.
"""


def ask_llm(question: str, df: pd.DataFrame, profile: dict, api_key: str) -> str:
    """
    Send a user question + dataset context to OpenAI.
    Returns the assistant's response as a string.
    """
    client = OpenAI(api_key=api_key)

    system_prompt = _build_system_prompt(df, profile)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # fast + cheap; swap to gpt-4o for deeper analysis
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "**Invalid API key.** Please check your OpenAI API key in the sidebar."
        elif "rate_limit" in error_msg.lower():
            return "**Rate limit hit.** Wait a moment and try again."
        elif "model" in error_msg.lower():
            return "**Model not available.** Check your OpenAI plan supports gpt-4o-mini."
        else:
            return f"**Error:** {error_msg}"
