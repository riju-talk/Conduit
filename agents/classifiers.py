from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()
# Assumes .env file contains a line: OPENAI_API_KEY=your-openai-api-key
# No need to set os.environ manually; ChatOpenAI will pick it up from the environment

few_shot_examples = [
    {
        "text": "Subject: Urgent complaint about order #12987\nI received the wrong item...",
        "format": "Email",
        "intent": "Complaint"
    },
    {
        "text": "{\n  \"product\": \"Widget A\",\n  \"quantity\": 100,\n  \"delivery_date\": \"2024-09-01\"\n}",
        "format": "JSON",
        "intent": "RFQ"
    },
    {
        "text": "Invoice #: 34897\nTotal Due: $5,204.99\nDue Date: 2024-08-10",
        "format": "PDF",
        "intent": "Invoice"
    },
    {
        "text": "According to Section 32 of the Data Privacy Act...",
        "format": "PDF",
        "intent": "Regulation"
    },
    {
        "text": "We suspect this invoice is fraudulent due to mismatched tax IDs...",
        "format": "Email",
        "intent": "Fraud Risk"
    },
]

example_prompt = PromptTemplate(
    input_variables=["text", "format", "intent"],
    template="Text: {text}\nFormat: {format}\nIntent: {intent}\n"
)

prefix = "Classify the given text into Format and Intent.\nChoose Format from [JSON, Email, PDF].\nChoose Intent from [RFQ, Complaint, Invoice, Regulation, Fraud Risk].\n\n"
suffix = "Text: {input_text}\nFormat:"

few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    prefix=prefix, 
    suffix=suffix,
    input_variables=["input_text"]
)

llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")  
chain = LLMChain(llm=llm, prompt=few_shot_prompt)

input_text = "Subject: Request for Quotation\nWe would like to receive a price estimate for 500 units of..."
result = chain.run(input_text=input_text)

import re
format_match = re.search(r"Format:\s*(\w+)", result)
intent_match = re.search(r"Intent:\s*(.*)", result)

classified_output = {
    "format": format_match.group(1) if format_match else "Unknown",
    "intent": intent_match.group(1).strip() if intent_match else "Unknown"
}

