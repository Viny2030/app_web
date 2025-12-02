import json
import shutil
import os

# ❌ LÍNEA ORIGINAL: from dashboard_template import TEMPLATE
# ✅ LÍNEA CORREGIDA:
from template import TEMPLATE


def generar_dashboard(df, predicciones, reales, tipo_problema, resultados_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar dataset
    df.to_csv(f"{output_dir}/dataset.csv", index=False)

    # Generar archivo dashboard.py
    contenido = TEMPLATE.format(
        tipo_problema=tipo_problema,
        resultados_json=json.dumps(resultados_dict),
        pred=predicciones,
        real=reales
    )

    with open(f"{output_dir}/dashboard.py", "w", encoding="utf-8") as f:
        f.write(contenido)

    # Comprimir carpeta para subir a Backblaze
    shutil.make_archive(output_dir, "zip", output_dir)

    return f"{output_dir}.zip"