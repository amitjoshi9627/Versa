## ✨ Versa: Your Personal AI Companion ✨

### Relaxed conversations, Real results!

#### Overview

_Versa_ is a user-friendly chatbot designed for versatile interactions, offering:

* **Relaxed Conversation:** Engage in natural, comfortable dialogues with _Versa_, feeling at ease to express yourself.
* **Real Results:** Get assistance through _Versa_'s ability to adapt to your specific requests and deliver tangible outcomes.

**Features**

* **Multi-Personality:** _Versa_ seamlessly transitions between various personalities, be it a therapist, expert, comedian, or a child. It tailors its interactions to your needs and preferences. (**Note:** Implementation details might vary depending on the current codebase.)
* **Intelligent Assistant:** Leverage _Versa_'s capabilities for document analysis, question answering, and general conversation, fostering a productive and efficient experience.
* **Humor and Fun:** Infuse levity into your interactions with _Versa_'s playful and witty personality. It can be a source of amusement and lighthearted engagement.
* **Adaptability:** Whether you seek a serious discussion, a lighthearted exchange, or a creative exploration, _Versa_ adapts its style and tone to accommodate your desires.

**Getting Started**

These instructions assume you have Git and Python installed on your system.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amitjoshi9627/Versa.git
   ```
2. **Install Dependencies:**

    Via `Conda`
   ```bash
   cd Versa
   conda env create -f environment.yml
   ```
   or using `pip`
   ```bash
   cd Versa
   pip install -r requirements.txt
   ```
3. **Run chatbot:**
   ```bash
   python -m examples.run_chatbot
   ```

4. **Run docbot:**
To run the module as a script, navigate to the root folder and run:
   ```bash
   python -m examples.run_docbot
   ```

5. **Run the streamlit app:**
To run the module as a script, navigate to the root folder and run:
   ```bash
   python -m streamlit run chatbot/streamlit/Home.py
   ```

**Usage**

Interact with _Versa_ using natural language prompts. Simply type your questions, requests, or topics of interest, and _Versa_ will respond accordingly.
**Note:** For information on Streamlit App - visit it's [Readme](chatbot/streamlit/README.md).
Similarly for API examples visit the [API Readme](examples/README.md)

**Contributing**

**We welcome contributions to Versa!**

To contribute to the project, please follow these steps:

**Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

1. **Fork the repository.**
2. **Create a new branch** for your feature or bug fix.
3. **Make your changes** and ensure code quality and consistency.
4. **Write tests** to cover your changes.
5. **Submit a pull request** clearly outlining the changes made.

**Additional Guidelines:**

* **Coding style:** Follow the OOPS coding style guidelines and write clean code.
* **Documentation:** Update documentation to reflect your changes. Include Docstrings and type annotations.
* **Formatting:** pre-commit automatically formats the code and has linter to ensure code correctness.
* **Testing:** Write comprehensive tests to ensure code quality.

We appreciate your contributions to the Versa project!

## Code of Conduct

**Our Pledge**

In the interest of fostering an open and welcoming environment, we pledge to make participation in our project and community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual orientation.

**Our Standards**

Examples of behavior that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery
* Personal attacks
* Trolling, insulting/derogatory comments, and other disruptive behavior
* Publicizing private information of others
* Other conduct which could reasonably be considered inappropriate in a professional setting

**Our Responsibilities**

Project maintainers are committed to fostering a welcoming and inclusive environment for all participants and will take appropriate and fair disciplinary measures to address any unacceptable behavior.

**Enforcement**

Instances of abusive, harassing, or otherwise unacceptable behavior may be subject to removal from participation in the project.

**Contact**

If you experience any issues or concerns with the project or its community, please contact [_@amitjoshi9627_].

We believe that a diverse and inclusive community is essential for building a successful and sustainable project. By following these guidelines, we can create a positive and welcoming environment for everyone.



**License**

_Versa_ is licensed under the GNU Affero General Public License (AGPL) (see [LICENSE](LICENSE) for details).

**Roadmap**

* **Construction in Progress** Constructing future plans for _Versa_, such as new features, improvements, or integrations.
