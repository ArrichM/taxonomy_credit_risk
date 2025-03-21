from typing import List, Dict, Literal
import os

import pandas as pd
import chromadb
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
import pydantic

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/max/.keys/done-diligence/done-diligence-60b76c0ed9c5.json"

class Dnsh(pydantic.BaseModel):
    climate_adaption: str
    climate_mitigation: str
    water: str
    circular_economy: str
    biodiversity: str
    pollution_prevention: str

class Dimension(pydantic.BaseModel):
    name: str
    description: str
    contribution: str
    dnsh: Dnsh

class Activity(pydantic.BaseModel):
    id: str
    title: str
    dimensions: List[Dimension]


class DimensionCompliance(pydantic.BaseModel):
    dimension_name: str = pydantic.Field(
        description="Name of the dimension, i.e. climate_adaption, climate_mitigation, water, circular_economy, biodiversity, pollution_prevention")
    reasoning_eligibility: str = pydantic.Field(description="Reasoning for the eligibility decision")
    is_eligible: bool = pydantic.Field(description="Is the company eligible in this dimension?")
    reasoning_alignment: str = pydantic.Field(description="Reasoning for the alignment decision")
    alignment: Literal["aligned", "not_aligned", "partially_aligned"] = pydantic.Field(
        description="Alignment status in this dimension")
    score: float = pydantic.Field(description="Score from 0 to 100 indicating alignment quality")
    dnsh_violated: bool = pydantic.Field(description="Does the company violate any DNSH criteria in this dimension?")


class Compliance(pydantic.BaseModel):
    dimensions: List[DimensionCompliance] = pydantic.Field(
        description="One assessment for each provided dimension of the requirements")


def get_dnsh(row, name: str) -> str:

    try:
        field = row[name]
    except KeyError:
        return "Not applicable"

    if pd.isna(field):
        return"Not applicable"

    return field


class TaxonomyClient:

    def __init__(self, activities: Dict[str, Activity]):

        self.chroma_client = chromadb.Client()
        self.activities_collection = self.chroma_client.create_collection(name="taxonomy_activities")
        self.activities = activities

        descriptions = {}
        for activity in activities.values():
            for dimension in activity.dimensions:
                descriptions[activity.id + ":" + dimension.name] = dimension.description

        self.activities_collection.add(
            documents=list(descriptions.values()),
            ids=list(descriptions.keys())
        )

        self.model = ChatVertexAI(
            model="gemini-2.0-flash-001",
            project="done-diligence",
            max_tokens=8192,
            location="europe-west1",
        )

    @classmethod
    def from_path(cls, taxonomy_path: str):

        taxonomy_sheets = pd.read_excel(taxonomy_path, sheet_name=None)

        all_activities = {}
        for sheet in taxonomy_sheets:
            local_sheet = taxonomy_sheets[sheet]
            for _, row in local_sheet.iterrows():
                activity_id = row["Activity"].lower().replace(" ", "_")

                dimension = Dimension(
                    name=sheet.lower().replace(" ", "_"),
                    description=row["Description"],
                    contribution=row["Substantial contribution criteria"],
                    dnsh=Dnsh(
                        climate_adaption=get_dnsh(row, "DNSH on Climate adaptation"),
                        climate_mitigation=get_dnsh(row, "DNSH on Climate mitigation"),
                        water=get_dnsh(row, "DNSH on Water"),
                        circular_economy=get_dnsh(row, "DNSH on Circular economy"),
                        biodiversity=get_dnsh(row, "DNSH on Biodiversity"),
                        pollution_prevention=get_dnsh(row, "DNSH on Pollution prevention"),
                    )
                )

                try:
                    activity = all_activities[activity_id]
                    activity.dimensions.append(dimension)
                except KeyError:
                    all_activities[activity_id] = Activity(
                        id=activity_id,
                        title=row["Activity"],
                        dimensions=[dimension],
                    )

        return cls(all_activities)


    def classify_activity(self, text: str, retrieval_cutoff: float = 1.65) -> Activity:

        result = self.activities_collection.query(query_texts=[text], n_results=20)

        activities = {
            activity_id: desc for activity_id, desc, distance in zip(
                result["ids"][0], result["documents"][0], result["distances"][0]
            ) if distance < retrieval_cutoff
        }

        agg_activities = {}
        for composite_activity_id, description in activities.items():
            activity_id = composite_activity_id.split(":")[0]
            dimension_id = composite_activity_id.split(":")[1]
            agg_activities[activity_id] = agg_activities.setdefault(activity_id, "") + f"\n{dimension_id}: " + description

        activities_block = ""
        for activity_id, description in agg_activities.items():
            activities_block += f"\n{'='*40}\nActivity ID:'{activity_id}':\n{description}"

        system_prompt = """
You will be provided with a list of activities and the text from the homepage of a company. 
Your task is to classify the company's main economic activity into one of the provided activities. 
Respond with the activity id and the reasoning that supports your classification.
If you are not sure about the classification, you can respond with "unknown".
"""
        human_message = f"""
Text of thw website:
{text}

Activities to classify:
{activities_block}

Now, please classify the main economic activity of the company based on the text.
"""
        class Classification(pydantic.BaseModel):
            reasoning: str
            activity_id: str

        resp = self.model.with_structured_output(Classification).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
        )

        classified_activity = self.activities[resp.activity_id]

        return classified_activity

    def classify_compliance(self, text: str, activity: Activity) -> Compliance:

        criteria_block = "".join([f"\n- {dim.name}" for dim in activity.dimensions])

        system_prompt = f"""
You are provided with two inputs:
1. The homepage text of a company.
2. A detailed set of sustainability and business eligibility criteria (copied above).

Your task is to assess the company’s performance in each of the following dimensions:
{criteria_block}

For each dimension, perform the following:

1. **Eligibility:**  
   - Determine if the company is potentially eligible, meaning if the business activities of the company could contribute to the criteria in that dimension.
   - Set "is_eligible" to true if the company is eligible in that dimension.

2. **Alignment:**  
   - Evaluate how well the company’s practices align with the criteria in the dimension.
   - Set "alignment" to "aligned", "partially_aligned", or "not_aligned". Here, aligned means that the company's practices are fully in line with the criteria, partially_aligned means that the company's practices are somewhat in line with the criteria, and not_aligned means that the company's practices are not in line with the criteria.
   - Provide a "score" between 0 and 100 to quantify the alignment quality.
   - Mark "dnsh_violated" as true if the company violates any Do No Significant Harm (DNSH) criteria in that dimension.

Analyze the provided texts carefully and output only the resulting JSON object.
"""

        human_message = f"""
Criteria:
{activity.model_dump_json(indent=1)}


Text of the website:
{text}
"""

        assessment = self.model.with_structured_output(Compliance).invoke(
            [
                SystemMessage(system_prompt),
                HumanMessage(human_message)
             ]
        )

        return assessment


taxonomy_path = "data/taxonomy.xlsx"

self = TaxonomyClient.from_path(taxonomy_path)

import requests
import trafilatura

resp = requests.get("https://www.homepowersolutions.de/en/product/")
text = trafilatura.extract(resp.text)

activity = self.classify_activity(text)

activity.title