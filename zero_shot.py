# Like all the other things we are going to import ollama
import ollama

# Lets read in the document that has been labeled
with open('labeling_demo.csv', 'r') as f:
    csv_file = f.read()

# The csv file is split up by new lines to seperate rows, and commas to seperate
# collums. We can use that to extract the 
# If you were to do this again, look into the package pandas as it has much
# better tools for handling tabulated data

## Use the split function to loop through the
sentences = []
labels = []
for row in csv_file.split('\n'):
    columns = row.split(',')
    if len(columns) > 1:
        sentences.append(columns[0])
        labels.append(columns[1])


