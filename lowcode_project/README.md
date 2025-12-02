# Proyecto Low-Code de Análisis de Datos

Este es un proyecto de análisis de datos diseñado para ser utilizado con herramientas de bajo código.

## Estructura del Proyecto

```
lowcode_project/
├── data/               # Datos en bruto y procesados
├── notebooks/          # Cuadernos de análisis
├── reports/            # Reportes y visualizaciones
└── src/                # Código fuente
    ├── app.py         # Aplicación principal
    ├── config.py      # Configuraciones
    └── utils.py       # Funciones de utilidad
```

## Requisitos

- Python 3.8+
- pip

## Instalación

1. Clona el repositorio
2. Crea un entorno virtual:
   ```
   python -m venv venv
   ```
3. Activa el entorno virtual:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar la aplicación:

```bash
streamlit run src/app.py
```

## Licencia

Este proyecto está bajo la Licencia MIT.
