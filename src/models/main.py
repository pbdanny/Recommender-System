from typing import List, Optional
from enum import IntEnum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

api = FastAPI()

class Priority(IntEnum):
    low = 1
    medium = 2
    high = 3

class TodoBase(BaseModel):
    name: str = Field(..., min_length=3, max_length=512, description="Name of the task", example="Build Recommender System API")
    description: str = Field(..., description="Detailed description of the task", example="Create an API to serve recommendations based on user data.")
    priority: Priority = Field(default=Priority.low, description="Priority level of the task", example=Priority.medium)

class TodoCreate(TodoBase):
    pass

class Todo(TodoBase):
    id: int = Field(..., description="Unique identifier for the task", example=1)

class TodoUpdate(TodoBase):
    name: Optional[str] = Field(None, min_length=3, max_length=512, description="Name of the task", example="Build Recommender System API")
    description: Optional[str] = Field(None, description="Detailed description of the task", example="Create an API to serve recommendations based on user data.")
    priority: Optional[Priority] = Field(None, description="Priority level of the task", example=Priority.medium)
    

all_todos = [
    Todo(id=1, name="Build Recommender System API", description="Create an API to serve recommendations based on user data.", priority=Priority.high),
    Todo(id=2, name="Implement calculation endpoint", description="Calculate recommendations using the trained model.", priority=Priority.medium),
    Todo(id=3, name="Test API endpoints", description="Run tests on all API endpoints.", priority=Priority.low),
    Todo(id=4, name="Deploy API to production", description="Deploy the API to the production environment.", priority=Priority.high),
    Todo(id=5, name="Monitor API performance", description="Monitor the API's performance in production.", priority=Priority.medium),
]

@api.get("/")
def index():
    return {"message": "Welcome to the Recommender System API"}

@api.get("/todos/{todo_id}", response_model=Todo)
async def get_todo(todo_id: int):
    for td in all_todos:
        if td.id == todo_id:
            return td

@api.get("/todos", response_model=List[Todo])
async def get_n_todos(n:int = None):
    if n:
        return all_todos[:n]
    else:
        return all_todos

@api.post("/todos", response_model=Todo)
async def create_todo(todo: TodoCreate):
    new_id = max(td.id for td in all_todos) + 1
    new_todo = Todo(id=new_id, name=todo.name, description=todo.description, priority=todo.priority)
    all_todos.append(new_todo)
    return new_todo

@api.put("/todos/{todo_id}", response_model=Todo)
async def update_todo(todo_id: int, updated_todo: TodoUpdate):
    for todo in all_todos:
        if todo.id == todo_id:
            todo.name = updated_todo.name or todo.name
            todo.description = updated_todo.description or todo.description
            todo.priority = updated_todo.priority or todo.priority
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

@api.delete("/todos/{todo_id}", response_model=Todo)
async def delete_todo(todo_id: int):
    for todo in all_todos:
        if todo.id == todo_id:
            all_todos.remove(todo)
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")