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

chat_bot = DocBotEngine(file_path=<your file path>)
question = "What is the typical size of a blue whale?"
response_message = chat_bot.get_response(query=question)
```

### Chat Bot

To use the general chatbot functionality, choose from the available `chatbot_type` options:

```python
from chatbot.engine import ChatBotEngine

# Available chatbot types: {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}
chat_bot = ChatBotEngine(chatbot_type=DEFAULT)
question = "What is 2/2?"
response_message = chat_bot.get_response(query=question)
```

### Chatbot Types:
The exact names of chatbot types are:
1. Therapist
2. Expert
3. Comedian
4. Child
5. Default

These can be imported from `chatbot.constants`

### Considerations

 - When using `Doc Bot` pasing `file_path` is mandatory.

### Response Format

The `get_response` method returns a `ResponseMessage` object with the following attributes:

* `query`: The original query.
* `response`: The generated response.


### License
This project is licensed under the GNU Affero General Public License (AGPL) (see it here - [LICENSE](../LICENSE) ).
