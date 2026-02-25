import logging

import numpy as np
from app.config import settings
from app.constants import NUMERIC_FEATURE_ORDER
from app.embedding_client import get_embeddings
from app.helpers import clean_text, compute_similarity
from app.schema import ResumeFullFeatures
from docling_core.types.doc.page import TextCellUnit
from docling_parse.pdf_parser import DoclingPdfParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

pdf_parser = DoclingPdfParser()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
)

SYSTEM_PROMPT = """ You are a strict structured data extractor. Return ONLY valid JSON.
No explanations. If missing, return 0. Follow scoring rules exactly."""

HUMAN_PROMPT = """
Extract the following:
1. Skills
2. Projects summary
3. Experience
4. Leadership experience

Scoring guidelines:

years_experience:
Estimate total years of professional experience.

current_title_seniority_score:
IC=1, Senior=2, Lead=3, Manager=4, Director/Head=5.

num_roles:
Count distinct professional positions.

top_tier_education_flag:
1 if IIT, IISc, NIT, PhD, or globally recognized top institution mentioned. Otherwise 0.

team_size_log:
If team size mentioned, compute log(team_size + 1). If unknown, 0.

mentorship_mentions_count:
Count explicit mentions of mentoring a team, leading a team/people, managing juniors.

architecture_ownership_score:
Count mentions of architected, designed, spearheaded, defined roadmap, owned system.

stakeholder_interaction_flag:
1 if executive, leadership, cross-functional collaboration mentioned. Else 0.

num_quantified_impacts:
Count metrics like %, $, reduction, increase.

max_percent_improvement:
Largest percentage improvement mentioned.

avg_percent_improvement:
Average of mentioned percentage improvements.

high_impact_flag:
1 if any impact > 20% or > $1M equivalent.

production_deployment_flag:
1 if deployed to production.

ci_cd_flag:
1 if CI/CD mentioned.

monitoring_flag:
1 if monitoring, drift detection, logging mentioned.

cloud_platform_count:
Count distinct cloud platforms (AWS, GCP, Azure).

llm_depth_score:
0=none
1=LLM API usage
2=RAG pipelines
3=multi-agent systems
4=fine-tuning

vector_db_flag:
1 if Pinecone, FAISS, Chroma, OpenSearch or any other vector database mentioned and
USED.

enterprise_scale_flag:
1 if large-scale, millions of users, enterprise systems mentioned.

cross_functional_projects_count:
Count projects involving multiple teams or stakeholders.

Resume:
{resume_text}
"""

structured_llm = llm.with_structured_output(ResumeFullFeatures)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)

chain = prompt | structured_llm


def extract_text_from_pdf(file_path: str) -> str:
    logger.info("Parsing PDF: %s", file_path)

    pdf_doc = pdf_parser.load(path_or_stream=file_path)

    words = []

    for _, page in pdf_doc.iterate_pages():
        for word in page.iterate_cells(unit_type=TextCellUnit.WORD):
            words.append(word.text)

    full_text = " ".join(words)

    logger.debug("Extracted text length=%s", len(full_text))
    return full_text


def extract_resume_features(file_path: str):
    logger.info("extract_resume_features: processing %s", file_path)

    resume_text = extract_text_from_pdf(file_path)
    clean = clean_text(resume_text)

    features = chain.invoke({"resume_text": clean})
    data = features.model_dump()

    numeric = np.array(
        [data[k] for k in NUMERIC_FEATURE_ORDER],
        dtype=float,
    ).reshape(1, -1)

    texts = [
        data.get("Leadership", "") or "",
        data.get("Projects_Summary", "") or "",
    ]

    logger.debug("Numeric features shape=%s", numeric.shape)

    return numeric, texts


async def preprocess(file_path: str) -> np.ndarray:
    logger.info("Preprocessing started for %s", file_path)

    numeric, texts = extract_resume_features(file_path)

    resume_embs = await get_embeddings(texts)
    logger.debug("Embeddings shape=%s", resume_embs.shape)

    similarity = compute_similarity(resume_embs)

    if similarity.ndim == 1:
        similarity = similarity.reshape(1, -1)

    final_features = np.hstack([numeric, similarity])

    logger.info("Final feature vector shape=%s", final_features.shape)

    return final_features
