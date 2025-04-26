from google import genai
from google.genai.types import HttpOptions
client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="How old is Philadelphia city and what is signifcant of the city?",
)
print(response.text)
