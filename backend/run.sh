#!/bin/bash
echo "Iniciando BACKEND..."
uvicorn backend:app --host localhost --port 8002 --reload