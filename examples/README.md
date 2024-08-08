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
from chatbot.engine import ChatbotEngine

chat_bot = ChatbotEngine(chatbot_type=DOCBOT, file_path=<your file path>)
question = "What is the typical size of a blue whale?"
response_message = chat_bot.get_response(query=question)
```

### Chat Bot

To use the general chatbot functionality, choose from the available `chatbot_type` options:

```python
from chatbot.engine import ChatbotEngine

# Available chatbot types: {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}
chat_bot = ChatbotEngine(chatbot_type=DEFAULT)
question = "What is 2/2?"
response_message = chat_bot.get_response(query=question)
```

### Chatbot Types:
The exact names of chatbot types are:
1. Docbot
2. Therapist
3. Expert
4. Comedian
5. Child
6. Default

These can be imported from `chatbot.constants`

### Considerations

 - When using `Doc Bot` pasing `file_path` is mandatory.
 - Do not pass `file_path` with other `chatbot_type`.

### Response Format

The `get_response` method returns a `ResponseMessage` object with the following attributes:

* `query`: The original query.
* `response`: The generated response.


### License
This project is licensed under the GNU Affero General Public License (AGPL) (see it here - [LICENSE](../LICENSE) ).
