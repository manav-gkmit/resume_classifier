from pydantic import BaseModel


class ResumeFullFeatures(BaseModel):
    years_experience: float
    current_title_seniority_score: int
    num_roles: int
    top_tier_education_flag: int
    team_size_log: float
    mentorship_mentions_count: int
    architecture_ownership_score: int
    stakeholder_interaction_flag: int
    num_quantified_impacts: int
    max_percent_improvement: float
    avg_percent_improvement: float
    high_impact_flag: int
    production_deployment_flag: int
    ci_cd_flag: int
    monitoring_flag: int
    cloud_platform_count: int
    llm_depth_score: int
    vector_db_flag: int
    enterprise_scale_flag: int
    cross_functional_projects_count: int

    Leadership: str
    Projects_Summary: str


class PredictResponse(BaseModel):
    probability: float
