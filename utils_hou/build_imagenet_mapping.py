import json
import requests
import re
import ast


url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'

response = requests.get(url)

def add_quotes_to_numbers(s):
    return re.sub(r'(\d+)', r'"\1"', s)

result_string = add_quotes_to_numbers(response.text)

result = ast.literal_eval(result_string)
print(result)

with open('label_text.json','w') as f:
    f.write(json.dumps(result))

prompts = list(result.values())

with open('./data/imagenet/text_prompts.txt','w') as f:
    for item in prompts:
        f.write(item+'\n')
