import ollama

# Read the few-shot examples from few_shot.csv
# if you are doing this after generateign, then change this to the generated csv
with open("few_shot.csv", "r") as f:
    few_shot_csv = f.read()

few_shot_sentences = []
few_shot_labels = []
for row in few_shot_csv.split("\n"):
    columns = row.split(",")
    if len(columns) > 1:
        few_shot_sentences.append(columns[0])
        few_shot_labels.append(columns[1])

# Prepare the few-shot examples as messages for the model
few_shot_messages = []
for sent, label in zip(few_shot_sentences, few_shot_labels):
    # Add user message with example sentence
    few_shot_messages.append({"role": "user", "content": sent})
    # Add assistant message with example label
    few_shot_messages.append({"role": "assistant", "content": label})

# Read the labeling_demo.csv as before
with open("labeling_demo.csv", "r") as f:
    csv_file = f.read()

sentences = []
labels = []
for row in csv_file.split("\n"):
    columns = row.split(",")
    if len(columns) > 1:
        sentences.append(columns[0])
        labels.append(int(columns[1]))

prompt = """
You are a sentiment analysis machine tasked with determining people's attitudes
towards wind farms. You will be given a sentence that will be someone's response
to the question "What do you think about wind farms?". 

Respond with a single digit, -1, 0 or 1.
-1 means negative sentiment
0 is a neutral sentiment
1 is a positive sentiment
"""


def query_sentence(prompt, sentence, model="qwen2.5:0.5b"):
    # Combine system prompt, few-shot examples, and current user sentence
    messages = (
        [{"role": "system", "content": prompt}]
        + few_shot_messages
        + [{"role": "user", "content": sentence}]
    )
    response = ollama.chat(
        model=model,
        messages=messages,
    )
    return response.message.content


answers = []
for sentence in sentences:
    answer = query_sentence(prompt, sentence)
    answers.append(answer)

print(f'{"Sentence":<80} {"True Label":<10} {"LLM Label":<10}')
for i in range(len(answers)):
    print(f"{sentences[i]:<80} {labels[i]:<10} {answers[i]:<10}")
