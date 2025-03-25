from typing import List, Dict
import os

import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
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
    eligibility: int = pydantic.Field(description="Eligibility status in this dimension")
    reasoning_alignment: str = pydantic.Field(description="Reasoning for the alignment decision")
    alignment: int = pydantic.Field(description="Alignment status in this dimension")
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

    def __init__(self, activities: Dict[str, Activity], chroma_client: chromadb.Client = None):

        self.activities = activities
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-small"
        )
        if not chroma_client:
            self.chroma_client = chromadb.PersistentClient(path="data/chroma")

            self.activities_collection = self.chroma_client.create_collection(
                name="taxonomy_activities",
                embedding_function=self.sentence_transformer_ef
            )

            descriptions = {}
            for activity in activities.values():
                descriptions[activity.id] = activity.dimensions[0].description

            self.activities_collection.add(
                documents=list(descriptions.values()),
                ids=list(descriptions.keys())
            )
        else:
            self.chroma_client = chroma_client
            self.activities_collection = self.chroma_client.get_collection(name="taxonomy_activities", embedding_function=self.sentence_transformer_ef)

        self.model = ChatVertexAI(
            model="gemini-2.0-flash-001",
            project="done-diligence",
            max_tokens=8192,
            temperature=0,
            location="europe-west1",
        )

    @classmethod
    def from_chromadb_client_path(cls, taxonomy_path: str, chroma_path: str = "data/chroma"):
        client = chromadb.PersistentClient(path=chroma_path)
        return cls.from_path(taxonomy_path=taxonomy_path, chroma_client=client)

    @classmethod
    def from_path(cls, taxonomy_path: str, chroma_client: chromadb.Client = None):

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

        return cls(all_activities, chroma_client=chroma_client)


    def get_activities_and_pages(self, pages: List[str], n_activities: int = 20):

        pages_client = chromadb.EphemeralClient()
        pages_collection = pages_client.create_collection(
            name="pages_temp",
            embedding_function=self.sentence_transformer_ef
        )
        pages_collection.add(documents=pages, ids=[str(i) for i in range(len(pages))])

        # Step 1: get the activities best matching for the company
        all_docs = self.activities_collection.get(include=["embeddings"])
        activity_embeddings = {idx: emb for idx, emb in zip(all_docs["ids"], all_docs["embeddings"])}

        activity_ids = activity_embeddings.keys()
        activity_embeddings = np.array(list(activity_embeddings.values()))
        query_results = pages_collection.query(query_embeddings=activity_embeddings, n_results=min(len(pages), 500))
        # query_results is a dict with fields ["ids", "distances", "documents"]
        activity_scores = np.array(query_results["distances"]).mean(axis=1)

        matching_activities = {
            activity_id: activity_scores[i] for i, activity_id in enumerate(activity_ids)
        }

        # Keep the best 20 activities with the lowest avg. distance
        best_activities = [a[0] for a in sorted(matching_activities.items(), key=lambda x: x[1])[:n_activities]]


        page_frames = []
        for activity_id, (distances, ids, documents) in enumerate(zip(
                query_results["distances"], query_results["ids"], query_results["documents"]
        )):
            activity_name = list(activity_ids)[activity_id]
            if activity_name not in best_activities:
                continue
            page_frames.append(pd.DataFrame({
                "distance": distances,
                "page_id": ids,
                "page_text": documents,
                "activity":activity_name
            }))

        page_frame = pd.concat(page_frames).reset_index()
        page_frame.sort_values("distance", inplace=True)
        page_frame.drop_duplicates("page_id", inplace=True, keep="first")

        return page_frame, [self.activities[activity_id] for activity_id in best_activities]

    def classify_activity(self, texts: List[str], activities: List[List[Activity]]) -> List[Activity]:

        inputs = []
        for text, activities_local in zip(texts, activities):
            activities_block = ""
            for activity in activities_local:
                activities_block += f"\n{'='*40}\nActivity ID:'{activity.id}':\n{activity.dimensions[0].description}\n\n"

            system_prompt = """
    You will be provided with a list of activities and the text from the homepage of a company. 
    Your task is to classify the company's main economic activity into one of the provided activities. 
    Respond with the activity id and the reasoning that supports your classification.
    If you are not sure about the classification, you can respond with "unknown". If none of the activities apply to the company, please respond with "none".
    """
            human_message = f"""
    Text of thw website:
    {text}
    
    Activities to classify:
    {activities_block}
    
    Now, please classify the main economic activity of the company based on the text.
    """

            input = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            inputs.append(input)

        class Classification(pydantic.BaseModel):
            reasoning: str
            activity_id: str

        response = self.model.with_structured_output(Classification).batch(
            inputs
        )

        activities = []
        for resp in response:
            if resp.activity_id in ["none", "unknown"]:
                activities.append(None)
                continue
            activities.append(self.activities[resp.activity_id])

        return activities

    def classify_compliance(self, texts: List[str], activities: List[Activity]) -> List[Compliance]:

        inputs = []

        for text, activity in zip(texts, activities):

            criteria_block = "".join([f"\n- {dim.name}" for dim in activity.dimensions])

            system_prompt = f"""
You are provided with two inputs:
1. The homepage text of a company.
2. A detailed set of sustainability and business eligibility criteria taken from the EU taxonomy ias a JSON object.

Your task is to assess the company’s performance in each of the following dimensions:
{criteria_block}

For each dimension, perform the following:

1. **Interpretation:**  
   - **Activity Description:** Use the "description" field to understand the scope of the economic activity. This explains which companies’ main activities are in scope (eligible) for this taxonomy activity.
   - **Contribution:** The "contribution" field clarifies the requirements for possible alignment. It describes the substantial contribution criteria that the company must meet to be considered aligned with the taxonomy activity.
   - **DNSH Criteria:** The "dnsh" object provides guidance on whether the activity meets Do No Significant Harm (DNSH) requirements for various dimensions. Use the relevant DNSH key for the dimension you are analyzing.

2. **Eligibility:**  
    - Confirm whether the taxonomy activity (as described in the "description") applies to the company’s main economic activity based on its homepage text. If not so, mark the company as ineligible.
    - If the activity does not apply, set "eligibility" to 0 and explain in "reasoning_eligibility".
    - Otherwise, assess how well the company’s activities match the taxonomy description using a 0–10 scale:
        0 – Unknown: The provided website text does not allow for a clear assessment of eligibility.
        1 – Completely Ineligible: No company activities match the taxonomy activities description.
        2 – Barely Mentioned: Only tangential or vague references hint at a possible eligible activity, without substantive evidence.
        3 – Marginally Eligible: A minor, unclear activity might qualify, but overall relevance is very limited.
        4 – Questionably Eligible: Some activities have elements of eligibility, yet descriptions are ambiguous and key criteria are not clearly met.
        5 – Borderline Eligible: Certain core activities could be seen as eligible, but significant doubts remain due to incomplete or unclear details.
        6 – Moderately Eligible: A mix of activities is described—some clearly matching eligible definitions while others remain uncertain.
        7 – Largely Eligible: Most major activities align with taxonomy eligibility; minor aspects may need further clarification.
        8 – Highly Eligible: Core activities are clearly defined as eligible, with solid supporting evidence.
        9 – Very Highly Eligible: The company offers detailed, unambiguous descriptions and documentation for nearly all activities as eligible.
        10 – Fully Eligible: All activities are unambiguously taxonomy-eligible, with comprehensive, robust evidence and clear compliance with every criterion.
    - Provide a brief explanation (reasoning_eligibility) that summarizes your assessment and cites key phrases from the homepage text.
    - Note that for deciding on the eligibility, only see if the company's activities match the description of the taxonomy activity. Here, you do not yet consider the alignment of the company's practices with the taxonomy criteria.

3. **Alignment:**  
    - Evaluate how well the company’s practices (as indicated on the homepage) meet the requirements of the taxonomy activity (as described in the "contribution" field).
    - Use a 0–10 scale for alignment:
        0 – Unknown: The provided website text does not allow for a clear assessment of alignment.
        1 – Not Aligned: No evidence of meeting any alignment criteria.
        2 – Negligible Alignment: Only incidental or vague mentions of relevant criteria.
        3 – Very Low Alignment: Minimal and weak evidence of alignment with key technical standards.
        4 – Limited Alignment: Some aspects of alignment are present, though major criteria are unmet or ambiguous.
        5 – Partial Alignment: Certain criteria are met, but significant gaps or inconsistencies remain.
        6 – Moderate Alignment: Core criteria are generally addressed with acceptable evidence, despite some deficiencies.
        7 – Considerable Alignment: Most technical criteria are met with solid supporting documentation.
        8 – High Alignment: Clear, robust evidence across nearly all criteria, with minor gaps only.
        9 – Very High Alignment: Nearly complete adherence to all technical standards with comprehensive evidence.
        10 – Fully Aligned: Exemplary and unambiguous compliance with every relevant criterion, fully supported by evidence.
    - Provide a brief explanation in "reasoning_alignment" that highlights supporting evidence (including direct quotations if possible) from the company text. 

4. **DNSH Violation Check:**  
   - Based on the "dnsh" field for the relevant dimension in the JSON, determine if the company's practices violate any Do No Significant Harm (DNSH) criteria.
   - If there is evidence of a violation in the relevant DNSH aspect, set "dnsh_violated" to true; otherwise, set it to false.

Analyze the provided texts carefully and output only the resulting JSON object.
"""

            human_message = f"""
    Criteria:
    {activity.model_dump_json(indent=1)}
    
    
    Text of the website:
    {text}
    """

            input = [
                SystemMessage(system_prompt),
                HumanMessage(human_message)
             ]
            inputs.append(input)

        assessments = self.model.with_structured_output(Compliance).batch(inputs)

        return assessments
