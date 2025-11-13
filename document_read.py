import ollama


# Write out the query llm function again
def query_llm(prompt, model="qwen2.5:0.5b"):
    """
    Given a 'prompt' and the name of a model,
    return the LLM's text response (uses ollama SDK).
    Because the model has a default, we don't need to be explicit in which model
    to use if you don't want to.
    """
    # Send the request to Ollama and get the response dictionary (a kind of
    # "named list").
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    # Return ONLY the LLM's textual answer from the response.
    return response.message.content


# Import PyPDF2 for PDF reading
import PyPDF2


# Using the PyPDF library, we can write a function that will read a Pdf and
# extract out all the text
def read_pdf_text(pdf_path):
    """
    Opens a PDF file from the provided path and returns all its text as one string.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


# ---- Main workflow ----

pdf_path = "example.pdf"  # Change this to your PDF file

# Use our function to read the PDF
pdf_text = read_pdf_text(pdf_path)

# Create a prompt that includes the PDF's contents
# You can stick strings together with + symbols to make it one big string
prompt = pdf_text + "\n\n" + "What is the name of the author of this document"

# Use query_llm (imported from hyperspecific.py) to get the LLM's answer!
llm_answer = query_llm(prompt)

print("Ollama LLM says:\n")
print(llm_answer)
