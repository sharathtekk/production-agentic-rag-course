"""Feedback client for handling user feedback with production-grade features."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from collections import defaultdict

from src.schemas.api.ask import FeedbackRequest, FeedbackResponse
from src.services.langfuse.client import LangfuseTracer

logger = logging.getLogger(__name__)


class FeedbackClient:
    """Handles user feedback with rate limiting, deduplication, and metrics."""

    def __init__(self, langfuse_tracer: LangfuseTracer, max_retries: int = 3, timeout_seconds: float = 10.0):
        """Initialize feedback client.

        Args:
            langfuse_tracer: LangfuseTracer instance for submitting feedback
            max_retries: Maximum number of submission retries
            timeout_seconds: Timeout for Langfuse submission
        """
        self.langfuse_tracer = langfuse_tracer
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Rate limiting: track feedback submissions per trace_id
        # Max 5 feedback submissions per trace_id per hour
        self._rate_limit_tracker: Dict[str, list] = defaultdict(list)
        self._rate_limit_max_per_hour = 5

        # Deduplication: track submitted feedback
        self._submitted_feedback: Dict[str, Tuple[float, Optional[str]]] = {}

        # Metrics
        self._metrics = {
            "total_submitted": 0,
            "total_failed": 0,
            "total_rate_limited": 0,
            "total_duplicates": 0,
            "total_errors": 0,
        }

    def _check_rate_limit(self, trace_id: str) -> bool:
        """Check if feedback submission is allowed under rate limits.

        Args:
            trace_id: The trace ID being submitted

        Returns:
            True if submission is allowed, False if rate limited
        """
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=1)

        # Clean old entries
        self._rate_limit_tracker[trace_id] = [
            ts for ts in self._rate_limit_tracker[trace_id]
            if ts > cutoff_time
        ]

        if len(self._rate_limit_tracker[trace_id]) >= self._rate_limit_max_per_hour:
            self._metrics["total_rate_limited"] += 1
            logger.warning(
                f"Rate limit exceeded for trace_id: {trace_id}. "
                f"Max {self._rate_limit_max_per_hour} submissions per hour allowed."
            )
            return False

        # Record this submission attempt
        self._rate_limit_tracker[trace_id].append(now)
        return True

    def _check_duplicate(self, trace_id: str, score: float, comment: Optional[str]) -> bool:
        """Check if identical feedback has been submitted for this trace.

        Args:
            trace_id: The trace ID
            score: The feedback score
            comment: The feedback comment

        Returns:
            True if this is a duplicate, False otherwise
        """
        if trace_id in self._submitted_feedback:
            prev_score, prev_comment = self._submitted_feedback[trace_id]
            # Check if score and comment are identical
            if prev_score == score and prev_comment == comment:
                self._metrics["total_duplicates"] += 1
                logger.info(f"Duplicate feedback detected for trace_id: {trace_id}")
                return True
        return False

    def _record_submission(self, trace_id: str, score: float, comment: Optional[str]) -> None:
        """Record feedback submission for deduplication.

        Args:
            trace_id: The trace ID
            score: The feedback score
            comment: The feedback comment
        """
        self._submitted_feedback[trace_id] = (score, comment)

    async def submit_feedback_with_retry(
        self, request: FeedbackRequest, request_id: str
    ) -> Tuple[bool, str, Optional[str]]:
        """Submit feedback with retry logic and timeout handling.

        Args:
            request: FeedbackRequest containing trace_id, score, and comment
            request_id: Request tracking ID for logging

        Returns:
            Tuple of (success, message, error_detail)
        """
        trace_id = request.trace_id
        logger.info(
            f"[{request_id}] Processing feedback for trace_id: {trace_id}, "
            f"score: {request.score}, comment_length: {len(request.comment or '')}"
        )

        # Validation checks
        if not self.langfuse_tracer:
            error_msg = "Langfuse tracer is not available"
            logger.error(f"[{request_id}] {error_msg}")
            self._metrics["total_errors"] += 1
            return False, error_msg, "SERVICE_UNAVAILABLE"

        # Rate limiting check
        if not self._check_rate_limit(trace_id):
            error_msg = f"Rate limit exceeded for trace_id: {trace_id}"
            logger.warning(f"[{request_id}] {error_msg}")
            return False, error_msg, "RATE_LIMITED"

        # Deduplication check
        if self._check_duplicate(trace_id, request.score, request.comment):
            logger.info(f"[{request_id}] Feedback was duplicate but accepting")
            self._record_submission(trace_id, request.score, request.comment)
            self._metrics["total_submitted"] += 1
            return True, "Feedback recorded successfully (duplicate ignored)", None

        # Submit with retry logic
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"[{request_id}] Attempt {attempt}/{self.max_retries} to submit feedback"
                )

                # Submit with timeout
                success = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.langfuse_tracer.submit_feedback,
                        trace_id,
                        request.score,
                        request.comment,
                    ),
                    timeout=self.timeout_seconds,
                )

                if success:
                    # Flush to ensure feedback is sent
                    await asyncio.to_thread(self.langfuse_tracer.flush)

                    self._record_submission(request.trace_id, request.score, request.comment)
                    self._metrics["total_submitted"] += 1

                    logger.info(
                        f"[{request_id}] Feedback submitted successfully for trace_id: {trace_id}"
                    )
                    return True, "Feedback recorded successfully", None

                elif attempt < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_seconds = 2 ** (attempt - 1)
                    logger.warning(
                        f"[{request_id}] Submission failed (attempt {attempt}). "
                        f"Retrying in {backoff_seconds}s..."
                    )
                    await asyncio.sleep(backoff_seconds)

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    backoff_seconds = 2 ** (attempt - 1)
                    logger.warning(
                        f"[{request_id}] Submission timeout (attempt {attempt}). "
                        f"Retrying in {backoff_seconds}s..."
                    )
                    await asyncio.sleep(backoff_seconds)
                else:
                    error_msg = (
                        f"Feedback submission timed out after {self.max_retries} attempts "
                        f"({self.timeout_seconds}s timeout)"
                    )
                    logger.error(f"[{request_id}] {error_msg}")
                    self._metrics["total_failed"] += 1
                    return False, error_msg, "TIMEOUT"

            except Exception as e:
                if attempt < self.max_retries:
                    backoff_seconds = 2 ** (attempt - 1)
                    logger.warning(
                        f"[{request_id}] Error during submission (attempt {attempt}): {str(e)}. "
                        f"Retrying in {backoff_seconds}s..."
                    )
                    await asyncio.sleep(backoff_seconds)
                else:
                    error_msg = f"Feedback submission failed: {str(e)}"
                    logger.error(f"[{request_id}] {error_msg}")
                    self._metrics["total_errors"] += 1
                    return False, error_msg, "INTERNAL_ERROR"

        # All retries exhausted
        error_msg = f"Failed to submit feedback after {self.max_retries} attempts"
        logger.error(f"[{request_id}] {error_msg}")
        self._metrics["total_failed"] += 1
        return False, error_msg, "MAX_RETRIES_EXCEEDED"

    def get_metrics(self) -> Dict:
        """Get feedback submission metrics.

        Returns:
            Dictionary containing feedback metrics
        """
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset feedback metrics."""
        self._metrics = {
            "total_submitted": 0,
            "total_failed": 0,
            "total_rate_limited": 0,
            "total_duplicates": 0,
            "total_errors": 0,
        }
