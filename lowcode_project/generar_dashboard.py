import json
import shutil
import os
import pandas as pd

from template import TEMPLATE


def generar_dashboard(df, predicciones, reales, tipo_problema, resultados_dict, output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Validar que el DataFrame no esté vacío
        if df is None or df.empty:
            raise ValueError("El DataFrame proporcionado está vacío o es None")

        # =========================================================
        # 💡 Añadir las predicciones al DataFrame
        # =========================================================
        if tipo_problema in ["clasificacion", "regresion", "clustering_o_anomalias"] and predicciones:
            # Renombramos la columna de predicciones según el tipo de problema
            col_name = "Cluster_ID" if tipo_problema == "clustering_o_anomalias" else "Prediccion_ML"

            # Aseguramos que 'predicciones' sea una serie o lista con la longitud correcta
            if len(predicciones) == len(df):
                # Convertimos las etiquetas de clúster a string para Plotly si es clustering
                df[col_name] = pd.Series(predicciones).astype(
                    str) if tipo_problema == "clustering_o_anomalias" else pd.Series(predicciones)

                # Si hay datos 'reales' (supervisado), también los guardamos
                if reales and len(reales) == len(df):
                    df["Target_Real"] = pd.Series(reales)
            else:
                print(f"Alerta: Las predicciones ({len(predicciones)}) no coinciden con la longitud del DF ({len(df)}).")
                return None
        # =========================================================

        # Guardar dataset (ahora con la columna Cluster_ID/Prediccion)
        df.to_csv(f"{output_dir}/dataset.csv", index=False)

        # Generar archivo dashboard.py
        print("Generando archivo dashboard.py...")
        try:
            # Crear el contenido del dashboard manualmente para evitar problemas con el formato
            contenido = TEMPLATE.replace('{tipo_problema}', str(tipo_problema)) \
                             .replace('{resultados_json}', json.dumps(resultados_dict, ensure_ascii=False))
            
            # Escribir el archivo
            with open(f"{output_dir}/dashboard.py", "w", encoding="utf-8") as f:
                f.write(contenido)
                
            print("✅ Archivo dashboard.py generado correctamente")
            
        except Exception as e:
            print(f"❌ Error al generar el dashboard: {str(e)}")
            raise

        # Crear archivos adicionales necesarios
        print("Creando archivos adicionales...")
        
        # requirements.txt
        with open(f"{output_dir}/requirements.txt", "w") as f:
            f.write("streamlit==1.24.0\npandas\nplotly\nnumpy\nmatplotlib")

        # .env para el puerto
        with open(f"{output_dir}/Dockerfile", "w") as f:
            f.write("""FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
""")

        # Comprimir la carpeta
        print("Comprimiendo resultados...")
        shutil.make_archive(output_dir, 'zip', output_dir)
        
        print(f"✅ Dashboard generado exitosamente en {output_dir}.zip")
        return f"{output_dir}.zip"
        
    except Exception as e:
        print(f"❌ Error al generar el dashboard: {str(e)}")
        raise