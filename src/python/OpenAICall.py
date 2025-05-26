from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "tell me what's the meaning of MBS in securities market"},
  ]
)

print(completion.choices[0].message);
