import asyncio
import os

from pydantic import BaseModel, Field

from .baselines import create_estimator
from ..structs import Estimate, Estimator
from gpt_researcher import GPTResearcher
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class EstimatePydantic(BaseModel):
    """Extracted estimate from the text."""

    lower: float = Field(description="The lower estimate provided in the text.")
    value: float = Field(description="The median estimate inferred from the text.")
    upper: float = Field(description="The upper estimate provided in the text.")
    unit: str = Field(description="Unit of measurement used for the estimates above.")


async def arun_gpt_researcher_estimator(model: str, question: str, report: str = None) -> Estimate:
    load_dotenv()

    if report is None:
        gpt_researcher = GPTResearcher(query=question, report_type="research_report")
        research = await gpt_researcher.conduct_research()
        report = await gpt_researcher.write_report()

    llm_model = ChatOpenAI(
        model=model,
        api_key=os.environ["GPT_RESEARCHER_LLM_API_KEY"],
        base_url=os.environ["GPT_RESEARCHER_LLM_BASE_URL"],
    )
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    'Extract the estimate for a question "{question}".',
                ),
                ("human", "{text}"),
            ]
        ).partial(question=question)
        chain = prompt | llm_model.with_structured_output(
            EstimatePydantic, method="json_schema"
        )
        return Estimate(**dict(await chain.ainvoke({"text": report})), reasoning_trace={"context" : research, "report" : report})
    
    except:
        print("Defaulting to Pydantic Output Parser.")
        parser = PydanticOutputParser(pydantic_object=EstimatePydantic)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        'Extract the estimate for a question "{question}". Wrap the'
                        " output in `json` tags\n{format_instructions}"
                    ),
                ),
                ("human", "{text}"),
            ]
        ).partial(
            format_instructions=parser.get_format_instructions(), question=question
        )

        chain = prompt | llm_model | parser
        return Estimate(**dict(await chain.ainvoke({"text": report})), reasoning_trace={"context" : research, "report" : report})


def run_gpt_researcher_estimator(model: str, question: str) -> Estimate:
    return asyncio.run(arun_gpt_researcher_estimator(model, question))


def gpt_researcher_estimator(
    model: str,
) -> Estimator:
    return create_estimator(
        f"gpt_researcher_estimator:{model}", run_gpt_researcher_estimator, model
    )
