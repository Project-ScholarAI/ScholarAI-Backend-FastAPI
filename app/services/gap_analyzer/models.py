"""
Data models for the Gap Analyzer service.
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal
from uuid import uuid4
from pydantic import BaseModel, Field

AnalysisMode = Literal["light", "deep"]

class GapAnalysisRequest(BaseModel):
    """Request model for gap analysis"""
    url: str = Field(..., description="URL of the seed paper to analyze")
    max_papers: int = Field(default=10, ge=5, le=20, description="Maximum papers to analyze")
    validation_threshold: int = Field(default=2, ge=1, le=5, description="Number of validation attempts per gap")
    analysis_mode: AnalysisMode = Field(default="deep", description="Analysis depth mode: 'light' for fast 2-3 min analysis, 'deep' for comprehensive 10-15 min analysis")


@dataclass
class ResearchGap:
    """Represents a potential research gap with validation tracking"""
    gap_id: str
    description: str
    source_paper: str
    source_paper_title: str
    validation_strikes: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    category: Optional[str] = None
    
    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = str(uuid4())[:8]


@dataclass
class PaperAnalysis:
    """Structured analysis of a research paper"""
    url: str
    title: str
    key_findings: List[str]
    methods: List[str]
    limitations: List[str]
    future_work: List[str]
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    
    
class ResearchFrontierStats(BaseModel):
    """Rich statistics about the expanding research frontier"""
    frontier_expansions: int = Field(..., description="Number of times frontier expanded with new research areas")
    research_domains_explored: int = Field(..., description="Unique research domains discovered")
    cross_domain_connections: int = Field(..., description="Interdisciplinary connections found")
    breakthrough_potential_score: float = Field(..., ge=0, le=10, description="AI-assessed breakthrough potential (0-10)")
    research_velocity: float = Field(..., description="Papers per minute analyzed")
    gap_discovery_rate: float = Field(..., description="Gaps discovered per paper analyzed")
    elimination_effectiveness: float = Field(..., description="Percentage of gaps successfully eliminated")
    frontier_coverage: float = Field(..., ge=0, le=100, description="Estimated coverage of research landscape (%)")

class ResearchLandscape(BaseModel):
    """Visual representation of the research landscape"""
    dominant_research_areas: List[str] = Field(..., description="Primary research areas discovered")
    emerging_trends: List[str] = Field(..., description="Emerging research trends identified")
    research_clusters: Dict[str, int] = Field(..., description="Research area clusters with gap counts")
    interdisciplinary_bridges: List[str] = Field(..., description="Cross-domain research opportunities")
    hottest_research_areas: List[Dict[str, Any]] = Field(..., description="Research areas with highest activity")

class ProcessMetadata(BaseModel):
    """Comprehensive metadata about the gap analysis process"""
    request_id: str
    total_papers_analyzed: int
    processing_time_seconds: float
    gaps_discovered: int
    gaps_validated: int
    gaps_eliminated: int
    search_queries_executed: int
    validation_attempts: int
    seed_paper_url: str
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Enhanced frontier statistics
    frontier_stats: ResearchFrontierStats
    research_landscape: ResearchLandscape
    
    # Performance metrics
    avg_paper_analysis_time: float = Field(..., description="Average time per paper analysis (seconds)")
    successful_paper_extractions: int = Field(..., description="Papers successfully extracted and analyzed")
    failed_extractions: int = Field(..., description="Papers that failed extraction")
    
    # AI Intelligence metrics  
    gemini_api_calls: int = Field(..., description="Total Gemini API interactions")
    llm_tokens_processed: int = Field(..., description="Estimated tokens processed by LLM")
    ai_confidence_score: float = Field(..., ge=0, le=100, description="AI confidence in gap analysis (%)")
    
    # Research quality indicators
    citation_potential_score: float = Field(..., ge=0, le=10, description="Estimated citation potential of gaps")
    novelty_index: float = Field(..., ge=0, le=10, description="Novelty index of discovered gaps")
    impact_factor_projection: float = Field(..., description="Projected impact factor for gap research")


class GapMetrics(BaseModel):
    """Rich metrics for a research gap"""
    difficulty_score: float = Field(..., ge=0, le=10, description="Research difficulty assessment (0-10)")
    innovation_potential: float = Field(..., ge=0, le=10, description="Innovation potential score (0-10)")  
    commercial_viability: float = Field(..., ge=0, le=10, description="Commercial application potential (0-10)")
    time_to_solution: str = Field(..., description="Estimated time to solve (e.g., '2-3 years')")
    funding_likelihood: float = Field(..., ge=0, le=100, description="Likelihood of securing research funding (%)")
    collaboration_score: float = Field(..., ge=0, le=10, description="Potential for collaborative research (0-10)")
    ethical_considerations: float = Field(..., ge=0, le=10, description="Ethical complexity score (0-10)")

class ResearchContext(BaseModel):
    """Research context and ecosystem information"""
    related_gaps: List[str] = Field(..., description="Related research gaps in the same area")
    prerequisite_technologies: List[str] = Field(..., description="Technologies needed to address this gap")
    competitive_landscape: str = Field(..., description="Current competitive research landscape")
    key_researchers: List[str] = Field(..., description="Key researchers in this area")
    active_research_groups: List[str] = Field(..., description="Research groups working on related problems")
    recent_breakthroughs: List[str] = Field(..., description="Recent breakthroughs that make this gap more viable")

class ValidatedGap(BaseModel):
    """A research gap that has been validated and enriched"""
    gap_id: str
    gap_title: str = Field(..., description="Concise, compelling title for the research gap")
    description: str = Field(..., description="Detailed technical description of the gap")
    source_paper: str = Field(..., description="Paper where gap was first identified")
    source_paper_title: str
    validation_evidence: str = Field(..., description="Why this gap survived validation and is likely genuine")
    potential_impact: str = Field(..., description="Transformative potential and real-world applications")
    suggested_approaches: List[str] = Field(..., description="Specific, actionable research methodologies")
    category: str = Field(..., description="Primary research domain")
    
    # Enhanced gap intelligence
    gap_metrics: GapMetrics
    research_context: ResearchContext
    
    # Validation history
    validation_attempts: int = Field(..., description="Number of validation attempts survived")
    papers_checked_against: int = Field(..., description="Papers analyzed to validate this gap")
    confidence_score: float = Field(..., ge=0, le=100, description="AI confidence this is a genuine gap (%)")
    
    # Research opportunity assessment
    opportunity_tags: List[str] = Field(..., description="Tags describing research opportunity type")
    interdisciplinary_connections: List[str] = Field(..., description="Connections to other research domains")
    industry_relevance: List[str] = Field(..., description="Relevant industries for this research")
    
    # Timeline and resources
    estimated_researcher_years: float = Field(..., description="Estimated person-years to address")
    recommended_team_size: str = Field(..., description="Recommended research team composition")
    key_milestones: List[str] = Field(..., description="Suggested research milestones")
    success_metrics: List[str] = Field(..., description="How to measure success in addressing this gap")


class ExecutiveSummary(BaseModel):
    """High-level executive summary of the research frontier analysis"""
    frontier_overview: str = Field(..., description="Executive summary of research frontier discovered")
    key_insights: List[str] = Field(..., description="Top insights from the analysis")
    research_priorities: List[str] = Field(..., description="Recommended research priorities")
    investment_opportunities: List[str] = Field(..., description="Commercial investment opportunities identified")
    competitive_advantages: List[str] = Field(..., description="Potential competitive advantages from these gaps")
    risk_assessment: str = Field(..., description="Risk assessment for pursuing these research directions")

class EliminatedGap(BaseModel):
    """Information about gaps that were eliminated during analysis"""
    gap_title: str
    elimination_reason: str
    solved_by_paper: str
    elimination_confidence: float = Field(..., ge=0, le=100)

class ResearchIntelligence(BaseModel):
    """Advanced research intelligence and insights"""
    eliminated_gaps: List[EliminatedGap] = Field(..., description="Gaps that were eliminated during analysis")
    research_momentum: Dict[str, float] = Field(..., description="Research momentum by area (papers/month)")
    emerging_collaborations: List[str] = Field(..., description="Potential research collaborations identified")
    technology_readiness: Dict[str, int] = Field(..., description="Technology readiness levels by area (1-9)")
    patent_landscape: Dict[str, int] = Field(..., description="Patent activity by research area")
    funding_trends: Dict[str, str] = Field(..., description="Funding trends in related areas")

class GapAnalysisResponse(BaseModel):
    """Complete response from gap analysis with comprehensive research frontier intelligence"""
    request_id: str
    seed_paper_url: str
    
    # Core results
    validated_gaps: List[ValidatedGap] = Field(..., description="Research gaps that survived validation")
    executive_summary: ExecutiveSummary
    process_metadata: ProcessMetadata
    research_intelligence: ResearchIntelligence
    
    # Analysis metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_version: str = Field(default="2.0", description="Gap analyzer version")
    ai_models_used: List[str] = Field(default=["gemini-2.5-flash"], description="AI models used in analysis")
    
    # Visual data for frontend
    visualization_data: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Data for visualizing research frontier (graphs, charts, networks)"
    )
    
    # Quality assurance
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality metrics for the analysis"
    )
    
    # Future recommendations
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps for researchers"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }