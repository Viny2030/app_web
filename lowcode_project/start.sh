#!/bin/bash

echo "======================================"
echo "🚀 Servicio Automático de Auditoría ML"
echo "======================================"

if [ "$MODE" = "server" ]; then
    echo "🌐 Modo servidor Streamlit"
    streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
else
    echo "🤖 Modo automático (pipeline ML + B2)"
    python test_run.py
fi
