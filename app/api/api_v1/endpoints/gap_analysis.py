"""
Gap Analysis API endpoints for the Autonomous Research Frontier Agent.
Provides async background processing with status tracking.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import logging

from app.services.gap_analyzer.models import GapAnalysisRequest
from app.services.gap_analyzer.background_processor import background_processor, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gap-analysis", tags=["Gap Analysis"])

class GapAnalysisSubmissionRequest(BaseModel):
    """Request model for submitting gap analysis"""
    url: HttpUrl
    max_papers: Optional[int] = 10
    validation_threshold: Optional[int] = 2
    analysis_mode: Optional[str] = "deep"  # "light" or "deep"

class GapAnalysisSubmissionResponse(BaseModel):
    """Response when submitting gap analysis"""
    job_id: str
    status: str
    message: str
    estimated_time_minutes: int

class JobStatusResponse(BaseModel):
    """Response for job status queries"""
    job_id: str
    status: str
    created_at: str
    progress_message: str
    url: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None
    result_file: Optional[str] = None

@router.post("/submit", response_model=GapAnalysisSubmissionResponse)
async def submit_gap_analysis(request: GapAnalysisSubmissionRequest):
    """
    Submit a new gap analysis job for background processing.
    
    The analysis will run asynchronously and you can check progress using the job_id.
    Frontend should:
    1. Call this endpoint to start analysis
    2. Poll /status/{job_id} to check progress
    3. Get results from /result/{job_id} when completed
    
    Args:
        request: Gap analysis parameters including paper URL
        
    Returns:
        Job submission confirmation with tracking ID
    """
    try:
        # Convert to internal request model
        gap_request = GapAnalysisRequest(
            url=str(request.url),
            max_papers=request.max_papers or 10,
            validation_threshold=request.validation_threshold or 2,
            analysis_mode=request.analysis_mode or "deep"
        )
        
        # Submit job for background processing
        job_id = await background_processor.submit_job(gap_request)
        
        # Estimate processing time based on analysis mode and parameters
        if gap_request.analysis_mode == "light":
            # Light analysis: 2-3 minutes
            base_time = 2.5
            paper_multiplier = 0.1  # Much faster per paper
        else:
            # Deep analysis: 10-15 minutes
            base_time = 10
            paper_multiplier = 0.8  # Original timing
            
        estimated_minutes = max(
            2 if gap_request.analysis_mode == "light" else 8, 
            base_time + (gap_request.max_papers * paper_multiplier)
        )
        
        logger.info(f"🎯 Gap analysis job {job_id} submitted for {request.url}")
        
        return GapAnalysisSubmissionResponse(
            job_id=job_id,
            status="submitted",
            message="Gap analysis job submitted successfully. Use the job_id to track progress.",
            estimated_time_minutes=int(estimated_minutes)
        )
        
    except Exception as e:
        logger.error(f"Failed to submit gap analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit gap analysis: {str(e)}"
        )

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the current status of a gap analysis job.
    
    Frontend should poll this endpoint every 5-10 seconds while job is running.
    
    Possible statuses:
    - pending: Job queued but not started
    - running: Job actively processing 
    - completed: Job finished successfully
    - failed: Job encountered an error
    
    Args:
        job_id: The job identifier returned from submit endpoint
        
    Returns:
        Current job status and progress information
    """
    try:
        status_info = background_processor.get_job_status(job_id)
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        return JobStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )

@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    """
    Get the complete analysis result for a completed job.
    
    Frontend should call this only after status shows 'completed'.
    Returns the full research frontier analysis with validated gaps.
    
    Args:
        job_id: The job identifier returned from submit endpoint
        
    Returns:
        Complete gap analysis result with all research intelligence
    """
    try:
        # First check if job exists and is completed
        status_info = background_processor.get_job_status(job_id)
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        if status_info["status"] == JobStatus.PENDING.value:
            raise HTTPException(
                status_code=202,  # Accepted but not ready
                detail="Job is still pending. Please check back later."
            )
        
        if status_info["status"] == JobStatus.RUNNING.value:
            raise HTTPException(
                status_code=202,  # Accepted but not ready
                detail=f"Job is still running. Progress: {status_info['progress_message']}"
            )
        
        if status_info["status"] == JobStatus.FAILED.value:
            raise HTTPException(
                status_code=500,
                detail=f"Job failed: {status_info.get('error', 'Unknown error')}"
            )
        
        # Get the actual result
        result = background_processor.get_job_result(job_id)
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Job completed but result file not found"
            )
        
        logger.info(f"📊 Gap analysis result retrieved for job {job_id}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job result: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job result: {str(e)}"
        )

@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_recent_jobs(limit: int = 20):
    """
    List recent gap analysis jobs with their status.
    
    Useful for frontend to show job history or admin monitoring.
    
    Args:
        limit: Maximum number of jobs to return (default: 20, max: 100)
        
    Returns:
        List of recent jobs with status information
    """
    try:
        # Limit the number of jobs returned
        limit = min(limit, 100)
        
        jobs = background_processor.list_jobs(limit)
        
        return [JobStatusResponse(**job) for job in jobs]
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )

@router.get("/health")
async def gap_analysis_health():
    """
    Health check endpoint for gap analysis service.
    
    Returns service status and operational metrics.
    """
    try:
        return {
            "status": "healthy",
            "service": "Autonomous Research Frontier Agent",
            "version": "2.0.0",
            "running_jobs": background_processor.running_jobs,
            "max_concurrent_jobs": background_processor.max_concurrent_jobs,
            "total_jobs": len(background_processor.jobs),
            "results_directory": str(background_processor.results_dir),
            "jobs_directory": str(background_processor.jobs_dir),
            "features": [
                "Async background processing",
                "Job status tracking", 
                "Result persistence",
                "Persistent job status storage",
                "Multi-omics gap analysis",
                "Bioinformatics support"
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running or pending gap analysis job.
    
    Useful if user wants to stop a long-running analysis.
    
    Args:
        job_id: The job identifier to cancel
        
    Returns:
        Cancellation confirmation
    """
    try:
        status_info = background_processor.get_job_status(job_id)
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        if status_info["status"] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job that is already {status_info['status']}"
            )
        
        # Mark job as failed to effectively cancel it
        if job_id in background_processor.jobs:
            job = background_processor.jobs[job_id]
            job.status = JobStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.progress_message = "Job cancelled"
            # Save cancelled status to persistent storage
            background_processor._save_job_status(job_id)
        
        logger.info(f"🚫 Gap analysis job {job_id} cancelled")
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )

@router.get("/info")
async def service_info():
    """
    Get information about the gap analysis service capabilities.
    """
    return {
        "service_name": "Autonomous Research Frontier Agent v2.0",
        "description": "Discovers and validates research gaps through intelligent literature analysis with async processing",
        "new_features": [
            "🚀 Async background processing",
            "📊 Real-time status tracking", 
            "💾 Persistent result storage",
            "🧬 Multi-domain support (bioinformatics, AI, etc.)",
            "⚡ Non-blocking frontend integration"
        ],
        "workflow": {
            "1_submit": "POST /submit with paper URL",
            "2_poll": "GET /status/{job_id} every 5-10 seconds", 
            "3_retrieve": "GET /result/{job_id} when status='completed'"
        },
        "supported_domains": [
            "Bioinformatics & Computational Biology",
            "Computer Vision & AI",
            "Machine Learning",
            "Precision Medicine",
            "Multi-omics Integration"
        ],
        "processing_phases": [
            "Phase 1: Seed paper analysis and gap extraction",
            "Phase 2: Expanding frontier with related literature search", 
            "Phase 3: Gap validation through solution discovery",
            "Phase 4: Final synthesis with research intelligence"
        ],
        "parameters": {
            "max_papers": {
                "description": "Maximum papers to analyze",
                "range": "3-20",
                "default": 10,
                "impact": "Higher values = more comprehensive but slower"
            },
            "validation_threshold": {
                "description": "Number of validation attempts per gap",
                "range": "1-5", 
                "default": 2,
                "impact": "Higher values = more rigorous validation"
            }
        }
    }