"""
Multi-source academic search orchestrator
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..academic_apis.clients import (
    SemanticScholarClient,
    ArxivClient,
    CrossrefClient,
    PubMedClient,
)

from .deduplication import PaperDeduplicationService
from .filter_service import SearchFilterService
from .ai_refinement import AIQueryRefinementService
from .config import SearchConfig

logger = logging.getLogger(__name__)


class MultiSourceSearchOrchestrator:
    """
    Orchestrates paper search across multiple academic sources.
    
    Coordinates API clients, deduplication, filtering, and AI refinement
    to provide comprehensive academic paper search capabilities.
    """
    
    def __init__(self, search_config: SearchConfig):
        self.config = search_config
        
        # Initialize services
        self.deduplication_service = PaperDeduplicationService()
        self.filter_service = SearchFilterService(search_config.recent_years_filter)
        self.ai_service: Optional[AIQueryRefinementService] = None
        
        # Initialize academic API clients
        self._init_api_clients()
        
        # Active sources (Google Scholar removed)
        self.active_sources = [
            "Semantic Scholar",
            "arXiv",
            "Crossref", 
            "PubMed"
        ]
        
        logger.info(f"🎯 \033[96mSearch orchestrator initialized with {len(self.active_sources)} active sources\033[0m")
    
    def _init_api_clients(self):
        """Initialize all academic API clients"""
        self.api_clients = {
            "Semantic Scholar": SemanticScholarClient(),
            "arXiv": ArxivClient(),
            "Crossref": CrossrefClient(),
            "PubMed": PubMedClient(),
        }
        
        logger.debug(f"📡 API clients initialized: {list(self.api_clients.keys())}")
    
    def set_ai_service(self, ai_service: AIQueryRefinementService):
        """Set the AI refinement service"""
        self.ai_service = ai_service
        logger.info("🤖 AI service configured for query refinement")
    
    async def search_papers(
        self,
        query_terms: List[str],
        domain: str,
        target_size: int
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-round paper search across all academic sources.
        
        Args:
            query_terms: List of search terms
            domain: Research domain for filtering
            target_size: Target number of papers to retrieve
            
        Returns:
            List of unique papers from all sources
        """
        search_start_time = time.time()
        logger.info(f"🚀 \033[92mStarting multi-source search for terms: {query_terms}\033[0m")
        logger.info(f"🎯 \033[92mTarget: {target_size} papers from {len(self.active_sources)} sources\033[0m")
        logger.info(f"🔬 Domain: {domain}")
        
        # Reset deduplication for new search
        self.deduplication_service.reset()
        
        # Start with original query
        search_queries = [" ".join(query_terms)]
        
        # Execute search rounds
        for round_num in range(self.config.max_search_rounds):
            round_start_time = time.time()
            logger.info(f"📡 \033[94mRound {round_num + 1}: Searching with {len(search_queries)} queries\033[0m")
            
            # Search with all queries in this round
            for query_idx, query in enumerate(search_queries):
                query_start_time = time.time()
                logger.info(f"🔍 \033[93mQuery {query_idx + 1}/{len(search_queries)}: '{query}'\033[0m")
                
                round_papers = await self._search_all_sources(query, domain)
                added_count = self.deduplication_service.add_papers(round_papers)
                
                query_duration = time.time() - query_start_time
                logger.info(f"✅ Query completed in {query_duration:.1f}s: {added_count} new papers added")
            
            # Check if target reached
            current_count = self.deduplication_service.get_paper_count()
            round_duration = time.time() - round_start_time
            logger.info(f"📊 Round {round_num + 1} completed in {round_duration:.1f}s: {current_count} total papers")
            
            if current_count >= target_size:
                logger.info(f"🎉 \033[92mTarget reached: {current_count} papers\033[0m")
                break
            
            # Generate refined queries for next round (if not last round)
            if round_num < self.config.max_search_rounds - 1:
                logger.info("🤖 Generating refined queries for next round...")
                refined_queries = await self._generate_refined_queries(
                    query_terms, domain, self.deduplication_service.get_papers()
                )
                
                if refined_queries:
                    search_queries = refined_queries
                    logger.info(f"🔍 Using {len(refined_queries)} refined queries for next round")
                else:
                    logger.info("❌ No refined queries generated, ending search")
                    break
            
        final_papers = self.deduplication_service.get_papers()[:target_size]
        total_duration = time.time() - search_start_time
        logger.info(f"🎉 \033[92mSearch completed in {total_duration:.1f}s: {len(final_papers)} papers collected\033[0m")
        
        return final_papers
    
    async def _search_all_sources(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """
        Search all active academic sources in parallel.
        
        Args:
            query: Search query string
            domain: Research domain for filtering
            
        Returns:
            Combined list of papers from all sources
        """
        parallel_start_time = time.time()
        logger.info(f"🌐 \033[96mSearching {len(self.active_sources)} sources in parallel...\033[0m")
        
        # Create search tasks for all active sources
        search_tasks = []
        for source_name in self.active_sources:
            task = self._safe_source_search(source_name, query, domain)
            search_tasks.append(task)
        
        # Execute all searches in parallel with progress tracking
        logger.info("⏳ Waiting for API responses...")
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine results from all sources
        all_papers = []
        for i, result in enumerate(results):
            source_name = self.active_sources[i]
            
            if isinstance(result, Exception):
                logger.warning(f"❌ \033[91m{source_name} search failed: {str(result)}\033[0m")
                continue
            
            if isinstance(result, list) and result:
                logger.info(f"✅ \033[92m{source_name}: {len(result)} papers\033[0m")
                # Add source metadata to papers
                for paper in result:
                    paper["source"] = source_name
                all_papers.extend(result)
            else:
                logger.info(f"ℹ️ \033[93m{source_name}: No papers found\033[0m")
        
        parallel_duration = time.time() - parallel_start_time
        logger.info(f"📊 \033[96mParallel search completed in {parallel_duration:.1f}s: {len(all_papers)} total papers\033[0m")
        return all_papers
    
    async def _safe_source_search(
        self, 
        source_name: str, 
        query: str, 
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Safely search a single academic source with error handling.
        
        Args:
            source_name: Name of the academic source
            query: Search query
            domain: Research domain
            
        Returns:
            List of papers from the source (empty list on error)
        """
        source_start_time = time.time()
        logger.info(f"🔍 \033[94mStarting {source_name} search...\033[0m")
        
        try:
            # Get API client for this source
            client = self.api_clients.get(source_name)
            if not client:
                logger.error(f"❌ No client found for source: {source_name}")
                return []
            
            # Build filters for this source
            logger.debug(f"🔧 Building filters for {source_name}")
            filters = self.filter_service.build_filters(source_name, domain, query)
            logger.debug(f"📋 {source_name} filters: {filters}")
            
            # Execute search with retry logic for rate limiting
            papers = None
            for attempt in range(self.config.max_rate_limit_retries + 1):
                try:
                    api_start_time = time.time()
                    logger.info(f"📡 \033[96mCalling {source_name} API (attempt {attempt + 1})...\033[0m")
                    
                    # Add timeout to detect slow APIs
                    papers = await asyncio.wait_for(
                        client.search_papers(
                            query=query,
                            limit=self.config.papers_per_source,
                            filters=filters
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    api_duration = time.time() - api_start_time
                    if api_duration > 10.0:
                        logger.warning(f"🐌 \033[93m{source_name} API was slow: {api_duration:.1f}s\033[0m")
                    else:
                        logger.info(f"⚡ {source_name} API responded in {api_duration:.1f}s")
                    
                    break  # Success, exit retry loop
                    
                except asyncio.TimeoutError:
                    api_duration = time.time() - api_start_time
                    logger.warning(f"⏱️ \033[91m{source_name} API timeout after {api_duration:.1f}s\033[0m")
                    return []
                    
                except Exception as e:
                    api_duration = time.time() - api_start_time
                    error_str = str(e).lower()
                    
                    if "rate limit" in error_str or "429" in error_str:
                        if attempt < self.config.max_rate_limit_retries:
                            wait_time = self.config.rate_limit_backoff_seconds
                            # Enhanced logging for rate limiting
                            logger.warning(
                                f"⚠️ \033[93mRate limited for {source_name}, attempt {attempt + 1}. "
                                f"⏳ Waiting {wait_time}s before retry...\033[0m"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning(
                                f"🚫 \033[91mMax rate limit retries exceeded for {source_name}. "
                                f"Skipping this source for now.\033[0m"
                            )
                            return []
                    else:
                        # Non-rate-limit error, don't retry
                        logger.error(f"❌ \033[91m{source_name} API error after {api_duration:.1f}s: {str(e)}\033[0m")
                        raise e
            
            source_duration = time.time() - source_start_time
            result_count = len(papers) if papers else 0
            logger.info(f"✅ \033[92m{source_name} completed in {source_duration:.1f}s: {result_count} papers\033[0m")
            
            return papers or []
            
        except Exception as e:
            source_duration = time.time() - source_start_time
            logger.error(f"💥 \033[91m{source_name} failed after {source_duration:.1f}s: {str(e)}\033[0m")
            return []
    
    async def _generate_refined_queries(
        self,
        original_terms: List[str],
        domain: str,
        found_papers: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate refined search queries using AI service.
        
        Args:
            original_terms: Original search terms
            domain: Research domain
            found_papers: Papers found so far for context
            
        Returns:
            List of refined query strings
        """
        if not self.ai_service or not self.ai_service.is_ready():
            logger.debug("AI service not available for query refinement")
            return []
        
        if not found_papers:
            logger.debug("No papers available for query refinement")
            return []
        
        try:
            # Use top papers for context (limit to avoid huge prompts)
            sample_papers = found_papers[:10]
            
            refined_queries = await self.ai_service.generate_refined_queries(
                original_terms=original_terms,
                domain=domain,
                sample_papers=sample_papers,
                max_queries=3
            )
            
            if refined_queries:
                logger.info(f"🤖 Generated {len(refined_queries)} refined queries")
                return refined_queries
            else:
                logger.info("🤖 No refined queries generated")
                return []
                
        except Exception as e:
            logger.error(f"Error in query refinement: {str(e)}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the current search session"""
        dedup_stats = self.deduplication_service.get_deduplication_stats()
        
        return {
            "active_sources": self.active_sources,
            "papers_per_source": self.config.papers_per_source,
            "max_search_rounds": self.config.max_search_rounds,
            "ai_enabled": self.ai_service.is_ready() if self.ai_service else False,
            **dedup_stats
        }
    
    async def close(self):
        """Clean up all resources"""
        logger.info("🔒 Closing search orchestrator...")
        
        # Close all API clients
        close_tasks = []
        for client in self.api_clients.values():
            if hasattr(client, 'close'):
                close_tasks.append(client.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Close AI service
        if self.ai_service:
            await self.ai_service.close()
        
        logger.info("✅ Search orchestrator closed") 