# Like all the other things we are going to import ollama
import ollama

# Lets read in the document that has been labeled
with open("labeling_demo.csv", "r") as f:
    csv_file = f.read()

# The csv file is split up by new lines to seperate rows, and commas to seperate
# collums. We can use that to extract the
# If you were to do this again, look into the package pandas as it has much
# better tools for handling tabulated data

# Use the split function to loop through the senetences. This will make 2 lists
# -- ordered elements that we can grab by position -- one for the sentence and
# then one for the labels
sentences = []
labels = []
# Don't worry about this, this is just reading in the file into a nicer way for
# us to deal with
for row in csv_file.split("\n"):
    columns = row.split(",")
    if len(columns) > 1:
        sentences.append(columns[0])
        labels.append(int(columns[1]))

# We are now going to make a prompt. My recomendation is to be hyper specific.
# Remeber that language models are good at following instructions but not good
# at working out what you want
prompt = """
You are a sentiment analysis machine tasked with determining people's attitudes
towards wind farms. You will be given a sentence that will be someone's response
to the question "What do you think about wind farms?". 

Respond with a single digit, -1, 0 or 1.
-1 means negative sentiment
0 is a neutral sentiment
1 is a positive sentiment
"""

# We can now use a for loop to see how the model responds to the senetences
# using the prompt
# For loops are usefull tools for itterating through a set of items -- like a
# list.


def query_sentence(prompt, sentence, model="qwen2.5:0.5b"):
    """
    Given a 'prompt' and the name of a model,
    return the LLM's text response (uses ollama SDK).
    Because the model has a default, we don't need to be explicit in which model
    to use if you don't want to.
    """
    # Send the request to Ollama and get the response dictionary (a kind of
    # "named list").
    # Note we are no
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": sentence},
        ],
    )
    # Return ONLY the LLM's textual answer from the response.
    return response.message.content


# Initialise an empty list to store our answers in later
answers = []
# To start the loop, we use the following syntax:
for sentence in sentences:
    # Inside the indent, we now have the sentence availiable, and we can do
    # whatever we want with it.
    # We are going to send the sentence to the llm, and then store the answer in
    # the ansers dict
    # Get the response from the moddel
    answer = query_sentence(prompt, sentence)
    # Add the response to the answer list
    answers.append(answer)

# This line prints the headers for the columns we will display
# "Sentence" will be left-aligned in a space 80 characters wide
# "True Label" will be left-aligned in a space 10 characters wide
# "LLM Label" will be left-aligned in a space 10 characters wide
print(f'{"Sentence":<80} {"True Label":<10} {"LLM Label":<10}')

# This starts a loop that goes through each index from 0 to the length of the
# 'answers' list minus one
for i in range(len(answers)):
    # For each index i, print the sentence, true label, and LLM label
    # sentences[i] is the sentence at position i, left-aligned in 80 spaces
    # labels[i] is the true label at position i, left-aligned in 10 spaces
    # answers[i] is the LLM label at position i, left-aligned in 10 spaces
    print(f"{sentences[i]:<80} {labels[i]:<10} {answers[i]:<10}")
