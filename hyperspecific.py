# --- What is a comment? ---
# A comment explains your code to humans but is ignored by the computer.
# In Python, a comment starts with the '#' symbol.
# Example: This is a comment.

# --- What are imports? ---
# An import brings in code from outside libraries so you can use it in your
# script.
# For example, we import ollama, a library that lets us communicate with local
# LLMs.

import ollama  # Now we can use the ollama library's features in our code

# --- What is a function? ---
# A function is a reusable block of code. You define one using 'def',
# give it a name, and it will run what you put inside when called.

# Let's define a function called 'query_llm' that will:
# - Take a prompt (the question or instruction for the LLM)
# - Use a named LLM model (such as 'llama2')
# - Return the LLM's response as text

def query_llm(prompt, model='qwen2.5:0.5b'):
    """
    Given a 'prompt' and the name of a model,
    return the LLM's text response (uses ollama SDK).
    Because the model has a default, we don't need to be explicit in which model
    to use if you don't want to. 
    """
    # Send the request to Ollama and get the response dictionary (a kind of
    # "named list").
    response = ollama.chat(model=model, messages=[{'role': 'user',
                                                        'content': prompt}])
    # Return ONLY the LLM's textual answer from the response.
    return response.message.content

# --- Using (calling) a function ---
# To use a function, write its name and provide any needed input ("arguments")
# in parentheses.

# Let's set up a question prompt for the LLM:
my_prompt = "What is the state of the art method for computational framining analysis?"

# Now, call our function with this prompt to get the LLM's answer:
llm_answer = query_llm(my_prompt)

# Finally, print out the LLM's answer for everyone to see!
print("Ollama LLM says:\n")
print(llm_answer)
