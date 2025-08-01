from fastapi import APIRouter

from app.api.api_v1.endpoints import admin, qa, gap_analysis, papercall

api_router = APIRouter()

# Include admin routes for B2 storage management
api_router.include_router(admin.router, prefix="/admin", tags=["Admin B2 Storage"])

# Include QA routes for paper question-answering
api_router.include_router(qa.router, tags=["Paper QA"])

# Include gap analysis routes for research frontier agent
api_router.include_router(gap_analysis.router, tags=["Gap Analysis"])

# Include PaperCall routes for conference calls and journal special issues
api_router.include_router(papercall.router, prefix="/papercall", tags=["PaperCall"])
