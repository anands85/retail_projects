from crewai import Agent, LLM
from crewai import Task, Crew
import json
import os
import pandas as pd
import numpy as np
from duckduckgo_search import DDGS

def load_data():
    company_names_DF = pd.read_excel('/Users/anand.srinivasan/Downloads/company_driven_modeling/company.xlsx')
    product_DF = pd.read_excel('/Users/anand.srinivasan/Downloads/company_driven_modeling/products.xlsx')
    return company_names_DF,products_DF

def load_llm(model):
    key = 
    base_url = 
    
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = base_url

    llm = LLM(
        model=model,
        api_key=key,
        base_url = base_url,
        temperature=0.0
    )
    return llm

def agentic_workflow(llm, input_data):
    # Tools defined
    from crewai.tools import BaseTool
    class InternetSearchTool(BaseTool):
        name: str = "DuckDuckGo Search Tool"
        description: str = "Search the web for a given query."

        def _run(self, query: str) -> list:
            """Search Internet for relevant information based on a query."""
            ddgs = DDGS()
            results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
            return results

    internet_search_tool = InternetSearchTool()

    # Setting up the agents
    research_company_agent = Agent(
        role = '''you are good and reliable information finder You are a simple company researcher. Your job is to find a company's website and get basic information about what they do.''',
        goal = 'Get top 5 search results for the company.',
        backstory = 'With experience in market research and company database you will search and retrieve the information.',
        verbose=False,
        cache=True,
        tools=[internet_search_tool]
    )

    research_company_products_agent = Agent(
        role = '''You focus only on the essentials: what does this company do and what do they sell or provide? i want just one summary for each company create a simple.''',
        goal = 'Get top 5 results for the company.',
        backstory = 'With experience in market research and company and products database you will search and retrieve the information.',
        verbose=False,
        cache=True,
        tools=[internet_search_tool]
    )

    research_company_summarizer_agent = Agent(
        role = '''You will focus on getting the best and common company URLs and description and summarize them.''',
        goal = 'Get the company name its URL and description.',
        backstory = 'With experience in market research and company and products database you will search and retrieve the information.',
        verbose=False,
        cache=True,
        llm=llm,
    )

    
    # Define tasks

    research_company = Task(
        description = '''Get all search results for {company_name} in {jurisdiction}. When given a company name, you:
                            1. Search for the company's official website
                            2. Find their main business description
                            3. Identify what products or services they offer''',
        expected_output = 'Return a JSON dictionary of web links, description, products services offered, and company details.',
        agent = research_company_agent,
        tools=[internet_search_tool]
    )

    research_company_products = Task(
        description='''Get the input company {company_name} and {products} in {jurisdiction} with focus only on the essentials: what does this company do and what do they sell or provide? i want just one summary for each company create a simple.''',
        expected_output='Get the dictionary of the search results with web links, description, products services offered, and company details', 
        agent=research_company_products_agent,
        tools=[internet_search_tool]
    )

    research_company_summarizer = Task(
        description='You will focus on getting the best and common company URLs and description and summarize them.',
        expected_output='Return a single dictionary with the company name, description, URL, products services offered. Desscription should not have the company name mentioned.',
        agent=research_company_summarizer_agent,
        context=[research_company, research_company_products]
    )

    # Assemble a crew with planning enabled
    crew = Crew(
        agents=[research_company_agent, research_company_products_agent, research_company_summarizer_agent],
        tasks=[research_company, research_company_products, research_company_summarizer],
        verbose=True,
        memory=True
    )

    # Execute tasks
    crew.kickoff(inputs=input_data)

    # Accessing the task output
    task_output = research_company_summarizer.output

    return task_output

def retrieve_output(task_output):    
    print(f"Task Description: {task_output.description}")
    print(f"Task Summary: {task_output.summary}")
    print(f"Raw Output: {task_output.raw}")
    output = task_output.raw
    if task_output.raw.find('```python')>-1:
        output = task_output.raw[task_output.raw.find('```python'):]
        output = output.replace('```python','').replace('```','').strip()
    else:
        output = task_output.raw[task_output.raw.find('```json'):]
        output = output.replace('```json','').replace('```','').strip()
    print(output)
    try:
        import ast
        output = ast.literal_eval(output)
    except:
        print("error parsing output dictionary")
    return task_output.description, task_output.summary, output
    
################

def main():
    # Get input data
    company_name_DF,products_DF = load_data()

    # Prepare the LLM
    model = 'gpt-4o'
    llm = load_llm(model)

    output_arr = []
    count = 0
    for company_name in company_name_DF['company_name'].values:
        count = count + 1
        if count>53:
            output_dict = {}
            products_filter_DF = products_DF[products_DF.company_name==company_name]
            input_data = {
                'company_name':company_name,
                'products' : list(products_filter_DF['title'].values),
                'jurisdiction' : 'USA'
            }
            task_output = agentic_workflow(llm,input_data)
            description,summary,output = retrieve_output(task_output)
            print('---------------------------------')
            print('Description: ', description)
            print('---------------------------------')
            print('Summary: ', summary)
            print('---------------------------------')
            print('Output: ', output)

            output_dict['company_name'] = company_name
            output_dict['output'] = output
            output_dict['desciption'] = description
            output_dict['summary'] = summary
            output_arr.append(pd.Series(output_dict))
            output_DF = pd.DataFrame(output_arr)
            output_DF.to_csv('company_description.csv')

if __name__ == '__main__':
    main()
