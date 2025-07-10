prompt = f'''

Below is the transcribed text of a user's speech query. Please provide a response to this question, which will
be converted to speech using TTS. Please follow these requirements for your response:

1. Your response should not contain content that cannot be synthesized by the TTS model, such as paren-
theses, ordered lists, etc. Numbers should be written in English words rather than Arabic numerals.

2. Your response should be very concise and to the point, avoiding lengthy explanations.
[instruction]: {instruction}
Please output in JSON format as follows: "response": response.
'''