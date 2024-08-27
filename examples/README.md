## ✨ Versa: Your Personal AI Companion ✨

### Relaxed conversations, Real results!

#### Versa's API

**Overview**

The Versa API provides programmatic access to the Versa chatbot, allowing developers to integrate its functionalities into their applications.

**Getting Started:**

**Using the API**

### Doc Bot

To use the Doc Bot functionality, set the `chatbot_type` to `DOCBOT` and provide the file path to the document:

```python
from chatbot.engine import DocBotEngine

doc_bot = DocBotEngine(
        file_path=<file_path>,
        model_name_or_path=<model_name_or_path>,
        quantize=<whether to quantize model>,
        llm_temp=<llm temperature>,
        access_token=<(optional) access_token>,
)
question = "What is the typical size of a blue whale?"
response_message = doc_bot.get_response(query=question)
print(
    f"{CHAT_SEPARATOR}Question: {response_message.query}{CHAT_SEPARATOR}Response: {response_message.response}"
)
```

### Chat Bot

To use the general chatbot functionality, choose from the available `chatbot_type` options:

```python
from chatbot.engine import ChatBotEngine

# Available chatbot types: {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}
chat_bot = ChatBotEngine(
        chatbot_type=<type of chatbot>,
        model_name_or_path=<model_name_or_path>,
        quantize=<whether to quantize model>,
        llm_temp=<llm temperature>,
        access_token=<(optional) access_token>,
)
question = "What is 2/2?"
response_message = chat_bot.get_response(query=question)
print(
    f"{CHAT_SEPARATOR}Question: {response_message.query}{CHAT_SEPARATOR}Response: {response_message.response}"
)
```

### Chatbot Types:
The exact names of chatbot types are:
1. Therapist
2. Expert
3. Comedian
4. Child
5. Default

These can be imported from `chatbot.constants`

### Arguments Available
**ChatBotEngine**:
1. `chatbot_type`: type of chatbot to be used.

**DocBotEngine**:
1. `file_path`: File path for processing.

**Common Arguments**:
1. `model_name_or_path`: model name or path.
2. `quantize`: whether to quantize model or not.
3. `llm_temp`:  Parameter influencing the balance between predictability and creativity in generated text (less than 1 for more deterministic or greater than 1 for more creative)
4. `access_token`: (Optional). Access token for gated models in huggingface

### Considerations

 - When using `Doc Bot`, passing `file_path` is mandatory.
 - If you are passing a gated model, please provide `access_token`.

### Response Format

The `get_response` method returns a `ResponseMessage` object with the following attributes:

* `query`: The original query.
* `response`: The generated response.


### License
This project is licensed under the GNU Affero General Public License (AGPL) (see it here - [LICENSE](../LICENSE) ).
