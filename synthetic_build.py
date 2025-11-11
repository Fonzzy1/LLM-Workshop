import ollama
import random
import csv

# Read the few-shot examples from few_shot.csv
with open("few_shot.csv", "r") as f:
    few_shot_csv = f.read()

few_shot_sentences = []
few_shot_labels = []
for row in few_shot_csv.split("\n"):
    columns = row.split(",")
    if len(columns) > 1:
        few_shot_sentences.append(columns[0])
        few_shot_labels.append(columns[1])

# Reverse labels and sentences for few-shot examples
few_shot_messages = []
for sent, label in zip(few_shot_sentences, few_shot_labels):
    few_shot_messages.append({"role": "user", "content": label})
    few_shot_messages.append({"role": "assistant", "content": sent})

prompt = """
You are a sentiment analysis machine tasked with generating sentences
with a given sentiment label about wind farms.

Given a sentiment label (-1, 0, 1), respond with a sentence that reflects
that sentiment about wind farms.
"""

labels_for_generation = [-1, 0, 1]


def query_label(prompt, label, model="qwen2.5:0.5b"):
    messages = (
        [{"role": "system", "content": prompt}]
        + few_shot_messages
        + [{"role": "user", "content": str(label)}]
    )
    response = ollama.chat(
        model=model,
        messages=messages,
    )
    return response.message.content.strip()


generated_examples = []

for _ in range(100):
    label = random.choice(labels_for_generation)
    sentence = query_label(prompt, label)
    generated_examples.append((label, sentence))

with open("generated_examples.csv", "w", encoding="utf-8") as f:
    for label, sentence in generated_examples:
        line = f'"{sentence.replace("\"", "\"\"")}",{label}\n'
        f.write(line)
