from crewai import Agent, LLM
from crewai import Task, Crew
import json
import os
#import openai


llm = LLM(
    model='gpt-4o',
    api_key=key,
    base_url = alfa_base_url
)

input_data = {
        'topic': {
            'company name': 'Garten Wellbeing, PBC',
            'product description': "NATURE'S BAKERY FIG/BLUEB 12ct,"
        }
    }


historical_taxcode_agent = Agent(
    role = "Research specialist in sales tax.",
    goal = "For a given merchant or company name get all the historical tax codes.",
    backstory = "With experience in natural language processing and sales tax, quickly access highly accurate, state-specific taxability rules and rates as well as cross-border trade content — eliminating manual research across multiple sources.",
    verbose = False,
    llm=llm
)

customer_similarity_taxcode_agent = Agent(
    role = "Senior research specialist in sales tax",
    goal = "For a given merchant or company name get all the historical tax codes to a related company name and product description",
    backstory = "With experience in natural language processing and sales tax, quickly access highly accurate, state-specific taxability rules and rates as well as cross-border trade content — eliminating manual research across multiple sources.",
    verbose = False,
    llm=llm
)

# Define tasks
research_historical = Task(
    description='Get the input company details lookup for all the existing Avalara tax codes mapped.',
    expected_output='Return a list of all the tax codes mapped to the company name.',
    agent=historical_taxcode_agent
)

research_new = Task(
    description='If the historical list of tax codes are empty, use the company details {input_data} to further identify the relevant Avalara tax codes.',
    expected_output='Return a list of all the tax codes similar to the company name and product description',
    agent=customer_similarity_taxcode_agent,
    context=[research_historical]

)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[historical_taxcode_agent, customer_similarity_taxcode_agent],
    tasks=[research_historical, research_new],
    verbose=True
    #knowledge_sources=[csv_source]
)

# Execute tasks

crew.kickoff()

# Accessing the task output
task_output = research_new.output

print(f"Task Description: {task_output.description}")
print(f"Task Summary: {task_output.summary}")
print(f"Raw Output: {task_output.raw}")
if task_output.json_dict:
    print(f"JSON Output: {json.dumps(task_output.json_dict, indent=2)}")
if task_output.pydantic:
    print(f"Pydantic Output: {task_output.pydantic}")