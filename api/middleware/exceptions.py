"""
Exception handlers for Peer Agent API.
"""

import logging
import uuid
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException

from models.schemas import ErrorResponse

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors."""
    logger.warning(f"Validation error for {request.url}: {exc.errors()}")
    
    error_response = ErrorResponse(
        error_type="ValidationError",
        error_message=f"Request validation failed: {exc.errors()[0]['msg']}",
        request_id=str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(mode='json')
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error for {request.url}: {exc.detail}")
    
    error_response = ErrorResponse(
        error_type="HTTPException",
        error_message=exc.detail,
        request_id=str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    import traceback
    
    logger.error(f"Unexpected error for {request.url}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    error_response = ErrorResponse(
        error_type="InternalServerError",
        error_message="An unexpected error occurred. Please try again later.",
        request_id=str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(mode='json')
    )


def register_exception_handlers(app: Any) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler) 