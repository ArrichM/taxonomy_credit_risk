from typing import List, Dict
import os

import pandas as pd
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
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

class Eligibility(pydantic.BaseModel):
    reasoning_eligibility: str = pydantic.Field(description="Reasoning for the eligibility decision")
    eligibility: int = pydantic.Field(description="Eligibility status in this dimension")


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
        self.sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-base"
        )
        if not chroma_client:
            self.chroma_client = chromadb.PersistentClient(path="data/chroma")

            self.activities_collection = self.chroma_client.create_collection(
                name="taxonomy_activities",
                embedding_function=self.sentence_transformer_ef
            )

            self.activities_collection.add(
                documents=["query: " + act.title for act in activities.values()],
                ids=list(activities.keys())
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
        try:
            pages_collection = pages_client.create_collection(
                name="pages_temp",
                embedding_function=self.sentence_transformer_ef
            )
            pages_collection.add(documents=["query: " + p for p in pages], ids=[str(i) for i in range(len(pages))])

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

        finally:
            pages_client.delete_collection("pages_temp")

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
            try:
                if resp.activity_id in ["none", "unknown"]:
                    activities.append(None)
                    continue
                activities.append(self.activities.get(resp.activity_id, None))
            except:
                activities.append(None)

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
    - Confirm whether the taxonomy activity (as described in the "description") applies to the company’s main economic activity based on its homepage text. If not so, mark the company as ineligible. Be careful here: closely read every aspect of the activities description according to the EU taxonomy.
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
        10 – Fully Eligible: All activities are unambiguously taxonomy-eligible.
    - Provide a brief explanation (reasoning_eligibility) that summarizes your assessment. Your reasoning must start with a verbatim citation from the description field of the taxonomy activity.
    - Note that for deciding on the eligibility, only see if the company's activities match the description of the taxonomy activity. Here, you do not yet consider the alignment of the company's practices with the taxonomy criteria.
    
**EXAMPLE FOR ELIGIBILITY**

Company Activity:
We are a leading research institute in the field of financial derivative pricing.
Activity Description:
Research, applied research and experimental development of solutions, processes, technologies, business models and other products dedicated to the reduction, avoidance or removal of GHG emissions (RD&I) for which the ability to reduce, remove or avoid GHG emissions in the target economic activities has at least been demonstrated in a relevant environment, corresponding to at least Technology Readiness Level (TRL) 6(384) ...[goes on]...

Eligibility Reasoning:
Citation: <[...] dedicated to the reduction, avoidance or removal of GHG emissions (RD&I) [...]> Reasoning: Although the company is a research institute, it has its focus on financial derivative pricing. The company's main activity is not related to the reduction, avoidance, or removal of GHG emissions as required in the activity description and achieving this would require a major change in the company's business model. Therefore, the company is not eligible for this taxonomy activity.
Eligibility Score:
eligibility: 1 -  Completely Ineligible

3. **Alignment:**  
    - Evaluate how well the company’s practices (as indicated on the homepage) meet the requirements as described in the "contribution" field of the dimension.
    - Use a 0–10 scale for alignment:
        0 – Unknown: The provided website text does not allow for a clear assessment of alignment.
        1 – Not Aligned: No evidence of meeting the contribution criteria.
        2 – Negligible Alignment: Only incidental or vague mentions of the contribution criteria.
        3 – Very Low Alignment: Minimal and weak evidence of alignment with the contribution criteria.
        4 – Limited Alignment: Some aspects of alignment are present, though major contribution criteria are unmet or ambiguous.
        5 – Partial Alignment: Certain contribution criteria are met, but significant gaps or inconsistencies remain.
        6 – Moderate Alignment: Core contribution criteria are generally addressed with acceptable evidence, despite some deficiencies.
        7 – Considerable Alignment: Most contribution criteria are met with solid supporting documentation.
        8 – High Alignment: Clear, robust evidence that the companies activities meet the contribution criteria.
        9 – Very High Alignment: Detailed, unambiguous descriptions and documentation for nearly all contribution criteria.
        10 – Fully Aligned: The contribution criteria are unambiguously met.
    - Provide a brief explanation in "reasoning_alignment". Your reasoning must start with a verbatim citation from the contribution field of the taxonomy activity.
    - NOTE: Alignment in one dimension does ONLY depend on the company meeting the requirements in the "contribution" field of the dimension. It does not depend on anything else, like being "green" or "sustainable" in any subjective way.

4. **DNSH Violation Check:**  
   - Based on the "dnsh" field for the relevant dimension in the JSON, determine if the company's practices violate any Do No Significant Harm (DNSH) criteria.
   - If there is evidence of a violation in the relevant DNSH aspect, set "dnsh_violated" to true; otherwise, set it to false.

**EXAMPLE FOR ELIGIBILITY**
Company Activity:
We develop comprehensive software for monitoring the Co2 emissions of your server farm.
Contribution Criteria:
Development or use of ICT solutions that are aimed at one of the following: a. running Ml models to optimize production processes towards lowering GHG emissions. b. collecting data enabling GHG emission reductions. c. Software development practices where development meets the criteria set out in Annex II of Regulation (EU) 2021/873. Such ICT solutions may include, inter alia, the use of decentralized technologies (i.e. distributed ledger technologies), Internet of Things (IoT), 5G and Artificial Intelligence.
Alignment Reasoning:
Citation: <Development or use of ICT solutions that are aimed at one of the following: [...] b. collecting data enabling GHG emission reductions.[...]>. Reasoning: The company develops software for monitoring CO2 emissions. This data can be seen as collecting data enabling GHG emission reductions. Since the contribution criteria explicitly requires only "one of the following", this is sufficient to meet the contribution criteria, the other ones are not required. Therefore, the company is aligned with the contribution criteria.
Alignment Score:
alignment: 9 - High Alignment

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

        assessments = self.model.with_structured_output(Compliance).batch(inputs, temperature=0)

        return assessments

    def classify_eligibility(self, texts: List[str], activities: List[Activity]) -> List[Eligibility]:

        inputs = []

        for text, activity in zip(texts, activities):
            criteria_block = "".join([f"\n- {dim.name}" for dim in activity.dimensions])

            system_prompt = """
You are provided with two inputs:
1. The description of a business activity of the EU taxonomy.
2. The text of the homepage of a company.

Your task is to closely read the description of the taxonomy activity and assess whether the company's main economic activity could be classified as eligible for this taxonomy activity.
Note that we do not care about a subjective notion of sustainability when answering the question of eligibility. The main question to answer is: Is the company's main economic activity in scope for this taxonomy activity?

**STEPS**
- Read the taxonomy activity description carefully.
- Based on the description, assess whether the company's main economic activity is eligible for this taxonomy activity. Eligibility means that the company's main economic activity may fall under the scope of the taxonomy activity without requiring a major change in the company's business model.
- If the activity does not apply, set "eligibility" to 1 and explain in "reasoning_eligibility".
Provide your answer on a 0-10 scale:
    0 – Unknown: The provided website text does not allow for an assessment of eligibility.
    1 – Completely Ineligible: No company activities match the taxonomy activities description.
    2 – Barely Mentioned: Only tangential or vague references hint at a possible eligible activity, without substantive evidence.
    3 – Marginally Eligible: A minor, unclear activity might qualify, but overall relevance is very limited.
    4 – Questionably Eligible: Some activities have elements of eligibility, yet descriptions are ambiguous and key criteria are not clearly met.
    5 – Borderline Eligible: Certain core activities could be seen as eligible, but significant doubts remain due to incomplete or unclear details.
    6 – Moderately Eligible: A mix of activities is described—some clearly matching eligible definitions while others remain uncertain.
    7 – Largely Eligible: Most major activities align with taxonomy eligibility; minor aspects may need further clarification.
    8 – Highly Eligible: Core activities are clearly defined as eligible, with solid supporting evidence.
    9 – Very Highly Eligible: The company offers detailed, unambiguous descriptions and documentation for nearly all activities as eligible.
    10 – Fully Eligible: All activities are unambiguously taxonomy-eligible.
    
- Provide a brief explanation (reasoning_eligibility) that summarizes your assessment. Your reasoning must start with a verbatim citation from the description field of the taxonomy activity. Use the following format for the reasoning:
    Citation: <[...] citation from the description field of the taxonomy activity, [...] possibly with omissions [...]> Reasoning: [Your reasoning here]

**EXAMPLE 1**
Taxonomy Activity Description:
Research, applied research and experimental development of solutions, processes, technologies, business models and other products dedicated to the reduction, avoidance or removal of GHG emissions (RD&I) for which the ability to reduce, remove or avoid GHG emissions in the target economic activities has at least been demonstrated in a relevant environment, corresponding to at least Technology Readiness Level (TRL) 6(384) ...[goes on]...

Website Text:
We are a leading research institute in the field of financial derivative pricing.

Output:
{
    "eligibility_reasoning": "Citation: < [...] applied research [...] dedicated to the reduction, avoidance or removal of GHG emissions (RD&I) [...]> Reasoning: Although the company is a research institute, it has its focus on financial derivative pricing. The company's main activity is not related to the reduction, avoidance, or removal of GHG emissions as required in the activity description and achieving this would require a major change in the company's business model. Also, the company's research can not be described as "applied". "Therefore, the company is not eligible for this taxonomy activity.",
    "eligibility": 1 
}

**EXAMPLE 2**
Taxonomy Activity Description:
Manufacture of medicinal products. The economic activities in this category could be associated with NACE code C21.2 in accordance with the statistical classification of economic activities established by Regulation (EC) No 1893/2006.

Website Text:
Locking compression technology by aap
aap’s patentierte LOQTEQ® Kerntechnologie vereint Fraktur- kompression und winkelstabile Verriegelung in einem OP-Schritt. Auf dieser Basis haben wir ein umfassendes Platten- und Schraubenportfolio geschaffen, das mehr als 95% aller relevanten Indikationen in der Traumachirurgie an den oberen und unteren Extremitäten abdeckt.
Herausragende Eigenschaften unserer LOQTEQ® Technologie:
- Variabel winkelstabile Frakturkompression von 0 bis 2mm
- Sichere und stabile Schrauben-Platten-Verbindung
- Minimiert den Effekt der Kaltverschweißung
Durch internationale Patente geschützt | Produkt bereits erfolgreich im Markt | Erteilte Zulassungen: CE, FDA, NMPA, Anvisa etc.

Output:
{
    "eligibility_reasoning": "Citation: <Manufacture of medicinal products. [...]> Reasoning: The company develops and manufactures screws and plates for fracture treatment. This qualifies as the manufacture of medicinal products as the products are used in medical procedures. The company's main activity is in scope for this taxonomy activity.", 
    "eligibility": 8
}

Now, Analyze the provided texts carefully and output only the resulting JSON object.
"""

            human_message = f"""
Taxonomy Activity Description:
{activity.dimensions[0].description}

Website Text:
{text}
"""

            input = [
                SystemMessage(system_prompt),
                HumanMessage(human_message)
            ]
            inputs.append(input)

        assessments = self.model.with_structured_output(Eligibility).batch(inputs, temperature=0)

        return assessments
