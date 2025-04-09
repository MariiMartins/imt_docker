#!/bin/bash
echo "Iniciando BACKEND..."
uvicorn backend:app --host localhost --port 8000 --reload
