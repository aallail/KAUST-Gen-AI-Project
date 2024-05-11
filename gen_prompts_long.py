from openai import OpenAI
import openai
import json
import backoff
import atexit
import time
# Your OpenAI API key
from httpx import HTTPStatusError, TimeoutException

def generate_background_modification():
    """Generate a modification instruction using the ChatGPT 3.5 API."""
    # Constructing the prompt based on the user's detailed requirements
    task = (
    "Background Modification Task: You are tasked with suggesting a single, concise instruction "
    "to modify a background scene from the Flintstones to convey a specific emotion. The "
    "description of the scene will be provided, along with one of five emotions: Anger, Fear, "
    "Disgust, Happiness, or Sadness.\n"
    "Instructions:\n"
    "Provide only one instruction for modifying the background. The instruction must be fewer "
    "than 10 words.\n"
    "The changes should significantly alter or add significant elements to the background.\n"
    "Your Task: Based on a given description and a specified emotion, provide your instruction "
    "for modifying the background to enhance that emotion. Remember, the changes should "
    "significantly alter or add significant elements to the background. Remember, only the "
    "instruction should be provided, with no prefixes or explanations. I will start providing "
    "descriptions and emotions."
    )
    client = OpenAI(api_key='sk-proj-wnvgscGs8ukd0tY6dR80T3BlbkFJWvTg2U3KPrMa97KmitzP')
    return client, task
@backoff.on_exception(backoff.expo, (openai.APIError, openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, HTTPStatusError, TimeoutException, Exception), max_tries=5)
def gen_image(messages, client):
    return client.chat.completions.create(
                model="gpt-3.5-turbo",  # Specify the correct model name
                messages=messages,
                max_tokens=50
            )
def process_scenes(scene_tuples):
    """Process each scene tuple to generate background modification instructions."""
    emotions = ["happiness", "sadness", "anger", "fear"]
    output = []
    def write_output():
        with open('output_long.json', 'w') as file:
            json.dump(output, file)

    atexit.register(write_output)  # Register the write_output function to be called on program exit

    i = 0
    client, task = generate_background_modification()
    for scene_tuple in scene_tuples:
#        if i%20 == 0:
#            client, prompt = generate_background_modification()
        description = scene_tuple[1]
        for emotion in emotions:
            #Generate the background modification instruction
            messages = [
                {"role": "system", "content": task},
                {"role": "user", "content": f"Description: {description}"},
                {"role": "user", "content": f"Emotion: {emotion}. Provide an instruction."}
            ]
            response = gen_image(messages, client)
            response_text = response.choices[0].message.content
            print(response_text)
            #response = "cock"
            dict = {"emotion": emotion, 
                    "description": scene_tuple[1], 
                    "file_name": scene_tuple[0], 
                    "response": response_text}
            output.append(dict)
            if i % 100 == 0:
                write_output()  # Write to the JSON file every 100 iterations
                if i % 1000:
                    print(i)
            i += 1
                #print(f"Description: {description}, Emotion: {emotion}, Instruction: {modification_instruction}")
    return output
    time.sleep(0.1)
# Example list of scene tuples
###Load JSON. 
with open('./flintstones_annotations_v2-0.json', 'r') as file:
    data = json.load(file)
list = []
for line in data:
    desc = line["description"]
    file_name = line["globalID"] 
    list.append((file_name, desc))


# Process the scenes
output = process_scenes(list)
print(output)
with open('output_long.json', 'w') as file:
    json.dump(output, file)