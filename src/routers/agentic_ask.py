import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from src.dependencies import AgenticRAGDep, LangfuseDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse
from src.services.feedback.client import FeedbackClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    try:
        result = await agentic_rag.ask(
            query=request.query,
        )

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=request.top_k,
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            trace_id=result.get("trace_id"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    http_request: Request,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This production-grade endpoint provides:
    - Input validation and sanitization
    - Rate limiting (5 submissions per trace_id per hour)
    - Deduplication of identical feedback
    - Retry logic with exponential backoff
    - Request tracing and detailed logging
    - Timeout handling
    - Comprehensive error reporting
    - Metrics collection

    Args:
        request: Feedback data including trace_id, score, and optional comment
        http_request: FastAPI request object for context
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse with success status, message, and tracking ID

    Raises:
        HTTPException: On validation errors, rate limit, or submission failures
    """
    # Generate request tracking ID for correlated logging
    request_id = f"fb-{uuid.uuid4().hex[:12]}"
    start_time = datetime.utcnow()

    try:
        logger.info(
            f"[{request_id}] Feedback endpoint called | "
            f"trace_id: {request.trace_id} | "
            f"client_ip: {http_request.client.host if http_request.client else 'unknown'}"
        )

        # Validate tracer availability
        if not langfuse_tracer:
            logger.error(f"[{request_id}] Langfuse tracer unavailable")
            raise HTTPException(
                status_code=503,
                detail="Langfuse tracing service is unavailable. Please try again later.",
            )

        # Initialize feedback client with production settings
        feedback_client = FeedbackClient(
            langfuse_tracer=langfuse_tracer,
            max_retries=3,
            timeout_seconds=10.0,
        )

        # Submit feedback with all production features
        success, message, error_code = await feedback_client.submit_feedback_with_retry(
            request=request,
            request_id=request_id,
        )

        # Calculate response time
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        if success:
            logger.info(
                f"[{request_id}] Feedback submission successful | "
                f"duration_ms: {duration_ms:.2f} | "
                f"metrics: {feedback_client.get_metrics()}"
            )

            return FeedbackResponse(
                success=True,
                message=message,
                request_id=request_id,
                timestamp=datetime.utcnow(),
                trace_id=request.trace_id,
            )

        # Handle submission failures with appropriate HTTP status codes
        if error_code == "SERVICE_UNAVAILABLE":
            logger.warning(f"[{request_id}] Service unavailable | duration_ms: {duration_ms:.2f}")
            raise HTTPException(
                status_code=503,
                detail=message,
            )

        elif error_code == "RATE_LIMITED":
            logger.warning(f"[{request_id}] Rate limit exceeded | duration_ms: {duration_ms:.2f}")
            raise HTTPException(
                status_code=429,
                detail=message,
                headers={"Retry-After": "3600"},  # Retry after 1 hour
            )

        elif error_code == "TIMEOUT":
            logger.warning(f"[{request_id}] Submission timeout | duration_ms: {duration_ms:.2f}")
            raise HTTPException(
                status_code=504,
                detail=message,
            )

        else:
            # Generic server errors
            logger.error(
                f"[{request_id}] Feedback submission failed | "
                f"error_code: {error_code} | "
                f"duration_ms: {duration_ms:.2f}"
            )
            raise HTTPException(
                status_code=500,
                detail=message,
            )

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feedback data: {str(e)}",
        )
    except Exception as e:
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.exception(
            f"[{request_id}] Unexpected error during feedback submission | "
            f"duration_ms: {duration_ms:.2f} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your feedback. "
                   "Please try again later.",
        )
