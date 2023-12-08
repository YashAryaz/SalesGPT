
import os
import re
# import your OpenAI key
from mykey import key,url

os.environ["OPENAI_API_KEY"] = key
import streamlit as st
from streamlit_chat import message
from typing import Any, Callable, Dict, List, Union

from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.prompts.base import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from pydantic import BaseModel, Field
from StageGeneration import StageAnalyzerChain
from SalesConversation import SalesConversationChain
from mongodb import MongoDBManager
import streamlit as st
from ingestion import setup_knowledge_base,get_tools
from openai import OpenAI
import json
import datetime
import logging

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# Define a Custom Prompt Template
prospect_data_gathering_prompt_template = """
As an AI sales assistant, your task is to extract valuable information from the conversation history between the sales agent and the prospect. This information includes:

1. What product or service is the prospect interested in?
2. When would the prospect prefer to be called back? 
3. Is the prospect interested in a demo? 
4. If yes, when would the prospect prefer to schedule the demo?
5. What is the prospect's budget?
6. Who are the decision-makers and what is their process?
7. What are the prospect's pain points?
8. Is the prospect considering other competitors? If yes, who are they?
9. What is the prospect's timeline for implementation or purchase?
10. Has the prospect used or is using a similar product or service? If yes, what has been their experience?

Please note that the conversation history is located between the first and second '==='.
===
{conversation_history}
===

Based on the conversation history, please answer the above questions. 

Please note that the current date and time is {date_time}.

Your answers should be formatted as follows:

    "interest": "product/service name",
    "preferred_callback_times": "YYYY-MM-DD HH:MM",
    "demo": "Yes/No",
    "demo_schedule": "YYYY-MM-DD HH:MM",
    "budget": "budget",
    "decision_makers_and_process": "description",
    "pain_points": "description",
    "competitors": "names",
    "timeline": "description",
    "previous_solutions_experience": "description"

If there is no conversation history, please return the JSON with all values as 'NA'.
Remember, your task is to extract information, not to add any extra details.
"""


model_name='gpt-4-1106-preview'
llm = ChatOpenAI(model=model_name,temperature=0.1)



    
class CustomPromptTemplateForTools(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


# Define a custom Output Parser


class SalesConvoOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"  # change for salesperson_name
    verbose: bool = False

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
            print("TEXT")
            print(f"Let check:{text}")
            print("-------")
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        
        if not match:
            ## TODO - this is not entirely reliable, sometimes results in an error.
           
            return AgentFinish(
                {
                    "output": text
                },
                text,
            )
            # raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "sales-agent"
    
SALES_AGENT_TOOLS_PROMPT = """
Never forget your name is {salesperson_name}. You work as a {salesperson_role}.
You work at company named {company_name}. {company_name}'s business is the following: {company_business}.
Company values are the following. {company_values}
You are contacting a potential prospect in order to {conversation_purpose}
Prospect name is {prospect_name}.
Your means of contacting the prospect is {conversation_type}
If you're asked about where you got the user's contact information, say that you got it from public records.
Keep your responses in short length to retain the user's attention. Never produce lists, just answers.
Start the conversation by just a greeting and how is the prospect doing without pitching in your first turn.
If the prospect requests a demo, schedule it at their convenience. If they express immediate interest in purchasing, offer two options:
1. Immediate purchase through the company website: www.AquaHome.com.
2. Schedule a callback with an expert for further assistance.
Do not generate names or any details about company or product which are not provided in the prompt.
Current date and time is {date_time}.
Always think about at which conversation stage you are at before answering:

1: Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Remember: Always clarify in your greeting the reason why you are calling.
2: Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.
3: Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.
4: Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.
5: Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.
6: Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.
7: Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.
8: End conversation: The prospect has to leave to call, the prospect is not interested, or next steps where already determined by the sales agent.

TOOLS:
------

{salesperson_name} has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tools}
Action Input: the input to the action, always a simple string input
Observation: the result of the action
```

If the result of the action is "I don't know." or "Sorry I don't know", then you have to say that to the user as described in the next sentence.
When you have a response to say to the Human, or if you do not need to use a tool, or if tool did not help, you MUST use the format:

```
Thought: Do I need to use a tool? No
{salesperson_name}: [your response here, if previously used a tool, rephrase latest observation, if unable to find the answer, say it]
```

You must respond according to the previous conversation history and the stage of the conversation you are at.
Only generate one response at a time and act as {salesperson_name} only!

Begin!

Previous conversation history:
{conversation_history}

{salesperson_name}:
{agent_scratchpad}
"""

class SalesGPT(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    current_conversation_stage: str = "1"
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)


    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = False

    conversation_stage_dict: Dict = {
        "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Don't forget to clarify in your greeting the reason why you are contacting the prospect.",
        "2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
        "3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
        "4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
        "5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
        "6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
        "7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
        "8": "End conversation: The prospect has to leave to call, the prospect is not interested, or next steps where already determined by the sales agent."
    }

    salesperson_name: str = "Max"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "AquaPure Solutions"
    company_business: str = "AquaPure Solutions is a leading water purlfier company that specializes in providing advanced RO water solutions. We offer a range of high-performance water purlfiers designed to deliver clean and purlfied water for a healthier lifestyle. Our products include state-of-the-art RO systems, water filters, and accessories to ensure the highest quality drinking water for our customers."
    prospect_name: str ="Sam"
    company_values: str = "At AquaPure Solutions, our mission is to enhance the well-being of our customers by providing them with the purest and safest drinking water. We understand the importance of clean water for a healthy life, and we are dedicated to delivering top-notch water purlfication solutions. Our commitment to excellence extends to both our products and the customer service we provide."
    conversation_purpose: str = "find out whether they are looking to achieve better water quality via installing an advanced RO water purlfier."
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    def determine_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history='"\n"'.join(self.conversation_history),
            current_conversation_stage=self.current_conversation_stage,
        )

        self.current_conversation_stage = self.retrieve_conversation_stage(
            conversation_stage_id
        )

        logging.info(self.current_conversation_stage)

    def determine_prospect_data(self):
        client = OpenAI()
        prompt = PromptTemplate(
            template=prospect_data_gathering_prompt_template,
            input_variables=["conversation_history"],
            )
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt.format(conversation_history='"\n"'.join(self.conversation_history),date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M %A"))},
            ]
            )
        try:
            prospect_data=json.loads(response.choices[0].message.content)
        except:
            logging.error("Error in parsing prospect data")
            prospect_data = {
                "interest": "NA",
                "preferred_callback_times": "NA",
                "demo": "NA",
                "demo_schedule": "NA",
                "budget": "NA",
                "decision_makers_and_process": "NA",
                "pain_points": "NA",
                "competitors": "NA",
                "timeline": "NA",
                "previous_solutions_experience": "NA"
            }
        db_manager = MongoDBManager(url, 'prospect')
        if db_manager.insert_json_into_collection('prospect_preference', prospect_data):
            logging.info("Inserted prospect data into MongoDB")
        else:
            logging.error("Failed to insert prospect data into MongoDB")





    def human_step(self, human_input):
        # process human input
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    def step(self):
        return self._call(inputs={})
        
    

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""

        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                prospect_name=self.prospect_name,
                company_values=self.company_values,
                date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M %A"),
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
            )

        else:
            ai_message = self.sales_conversation_utterance_chain.run(
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                prospect_name=self.prospect_name,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_history="\n".join(self.conversation_history),
                date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M %A"),
                conversation_stage=self.current_conversation_stage,
                conversation_type=self.conversation_type,
            )

        # Add agent's response to conversation history
        # print(f"{self.salesperson_name}: ", ai_message.rstrip("<END_OF_TURN>"))
        agent_name = self.salesperson_name
        ai_message = agent_name + ": " + ai_message
        if "<END_OF_TURN>" not in ai_message:
            ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)

        # print(ai_message)
        
        ai_message.replace("<END_OF_CALL>","")
        return ai_message.rstrip("<END_OF_TURN>")

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm, verbose=verbose
        )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is False:
            sales_agent_executor = None

        else:
            product_catalog = kwargs["product_catalog"]
            tools = get_tools(product_catalog)

            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    'prospect_name',
                    "company_values",
                    "conversation_purpose",
                    "date_time",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = SalesConvoOutputParser(ai_prefix=kwargs["salesperson_name"])

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                verbose=verbose,
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=verbose
            )

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            verbose=verbose,
            **kwargs,
        )
    # Set up of your agent

# Conversation stages - can be modified
conversation_stages = {
    "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
    "2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
    "3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
    "4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
    "5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
    "6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
    "7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
    "8": "End conversation: The prospect has to leave to call, the prospect is not interested, or next steps where already determined by the sales agent."
}

# Agent characteristics - can be modified
config = dict(
     salesperson_name="Max",
    salesperson_role="Business Development Representative",
    company_name="AquaPure Solutions",
    company_business="AquaPure Solutions is a leading water purlfier company that specializes in providing advanced RO water solutions. We offer a range of high-performance water purlfiers designed to deliver clean and purlfied water for a healthier lifestyle. Our products include state-of-the-art RO systems, water filters, and accessories to ensure the highest quality drinking water for our customers.",
    company_values="At AquaPure Solutions, our mission is to enhance the well-being of our customers by providing them with the purest and safest drinking water. We understand the importance of clean water for a healthy life, and we are dedicated to delivering top-notch water purlfication solutions. Our commitment to excellence extends to both our products and the customer service we provide.",
    prospect_name="Sam",
    conversation_purpose="find out whether they are looking to achieve better water quality via installing an advanced RO water purlfier.",
    conversation_history=[],
    conversation_type="call",
    conversation_stage=conversation_stages.get(
        "1",
        "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Always clarify in your greeting the reason why you are contacting the prospect.",
    ),
    use_tools=True,
    product_catalog="data.json",
)



if __name__ == "__main__":
    if 'sales_gpt' not in st.session_state:
        st.session_state['sales_gpt'] = SalesGPT.from_llm(llm, verbose=False, **config)
        # Seed the conversation
        st.session_state['sales_gpt'].seed_agent()
        st.session_state['sales_gpt'].determine_conversation_stage()
        logging.info("Initialized SalesGPT")


    

    if 'responses' not in st.session_state:
        st.session_state['responses'] = [] 
        st.session_state.responses.append(st.session_state['sales_gpt'].step())
        logging.info("Added initial response")

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    st.title("Sales GPT")
    response_container = st.container()
    textcontainer = st.container()
    
    with textcontainer:
        query = st.text_input("Query: ",value="", )
        end_conversation_button = st.button('End Conversation')
        if end_conversation_button:
                st.session_state['sales_gpt'].determine_prospect_data()
                logging.info("Ended conversation")

        if query and not end_conversation_button:
            with st.spinner("typing..."):
               st.session_state['sales_gpt'].human_step(query)
               response=st.session_state['sales_gpt'].step()
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
            logging.info(f"Added user query: {query}")
            logging.info(f"Added model response: {response}")
            
            
        

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
    



    

