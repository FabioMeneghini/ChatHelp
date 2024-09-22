# ChatHelp 🤖⚙️
_ChatHelp_ is a simple chat application that uses hybrid search algorithms and LLMs to answer questions about software programs based on the documentation file of the program in quesiton.

This application was developed as final project of the Bachelor's Degree in Computer Science at University of Padua during my internship period at _Zucchetti S.p.A._

## 📄 Documentation
The complete documentation of this project (requirements analisys, code design, testing and a deep mathematical analisys of the used algorithms) consists in my thesis, which can be found in this repo (`thesis.pdf`), in Italian.

## 🤓 Instructions
In order to run and use _ChatHelp_, make sure to follow all the steps below.

### Step 1 - Clone this repo
1. Clone this repo with the command `git clone https://github.com/FabioMeneghini/ChatHelp.git` to get your local copy

### Step 2 - Install Python and the required packages
1. Install `Python 3`
2. install the required libraries and packages with the command `pip install -r requirements.txt`

### Step 3 - Get a Groq API key
1. Create a GroqCloud account [here](https://console.groq.com/login)
2. create an API key [here](https://console.groq.com/keys)
3. copy your API key and paste it in the `.env` file in yor local copy of this repo

### Step 4 - Setup the database
1. install `PostgreSQL`
2. install the `pgvector` extension for `PostgreSQL` (follow the instructions [here](https://github.com/pgvector/pgvector))
3. create a database named `documentazione`
4. create a table named `docs` with the following fields: `codice` (smallint), `testo` (character varying, 4095), `vettore` (vector) and `sezione` (character varying, 255)
5. create a table named `file` with the field `nome` (character varying, 255)

Now the environment is set up correctly and the application is ready to be used.

### Step 5 - Run the application
1. Open the command prompt
2. go to your local repository location
3. run the application with the command `streamlit run src/gui_main_window.py`

### Step 6 - Upload a document
1. Access the advanced settings through the sidebar (the password is `admin`)
2. upload a document with the appropriate widget (only PDF and SAM files are supported)

### Step 7 - Use the application
1. Now you can use the application by asking something related to the program the document you uploaded refers to
