# Generate API Documentation from Code

**Target Capability:** Accurate OpenAPI/Swagger spec generation from a given codebase.

Tests whether a model can read code and produce correct, complete API documentation — a common real-world task.

---

## Prompt

```
Given the following FastAPI application code, generate a complete OpenAPI 3.0 specification (YAML format).

The spec must include:
- All paths with correct HTTP methods
- Request body schemas with types and constraints
- Response schemas for 200, 400, 404, and 422 status codes
- Security scheme (Bearer token) for protected endpoints
- Parameter descriptions and examples

Here is the application code:

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

app = FastAPI(title="Task Manager API", version="1.0.0")

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, example="Fix login bug")
    description: Optional[str] = Field(None, max_length=2000)
    priority: int = Field(..., ge=1, le=5, example=3)
    due_date: Optional[datetime] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    priority: Optional[int] = Field(None, ge=1, le=5)
    status: Optional[str] = Field(None, pattern="^(todo|in_progress|done|cancelled)$")
    due_date: Optional[datetime] = None

class TaskResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    priority: int
    status: str
    due_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime

class PaginatedTasks(BaseModel):
    items: list[TaskResponse]
    total: int
    page: int
    page_size: int

@app.post("/tasks", response_model=TaskResponse, status_code=201, summary="Create a new task")
def create_task(task: TaskCreate, token: str = Depends(verify_bearer_token)):
    pass

@app.get("/tasks", response_model=PaginatedTasks, summary="List tasks with pagination")
def list_tasks(page: int = 1, page_size: int = 20, status: Optional[str] = None, token: str = Depends(verify_bearer_token)):
    pass

@app.get("/tasks/{task_id}", response_model=TaskResponse, summary="Get a task by ID")
def get_task(task_id: int, token: str = Depends(verify_bearer_token)):
    pass

@app.patch("/tasks/{task_id}", response_model=TaskResponse, summary="Update a task")
def update_task(task_id: int, task: TaskUpdate, token: str = Depends(verify_bearer_token)):
    pass

@app.delete("/tasks/{task_id}", status_code=204, summary="Delete a task")
def delete_task(task_id: int, token: str = Depends(verify_bearer_token)):
    pass

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
```

Output ONLY the OpenAPI YAML. No explanation, no markdown wrapping.
```
