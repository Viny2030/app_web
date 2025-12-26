import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from src.services.backblaze_service import BackblazeService

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Subir a Backblaze B2",
    page_icon="☁️",
    layout="centered"
)

def main():
    st.title("Subir archivos a Backblaze B2")
    
    # Inicializar el servicio de Backblaze
    try:
        b2_service = BackblazeService()
        st.success("✅ Conexión exitosa con Backblaze B2")
    except Exception as e:
        st.error(f"❌ Error al conectar con Backblaze B2: {str(e)}")
        st.info("Por favor, verifica tus credenciales en el archivo .env")
        return

    # Sección de subida de archivos
    st.header("Subir archivo")
    uploaded_file = st.file_uploader("Selecciona un archivo para subir", type=None)
    
    # Selector de bucket
    bucket_name = st.selectbox(
        "Selecciona el bucket de destino",
        [os.getenv('B2_BUCKET_INPUT', 'datos-entrada'), 
         os.getenv('B2_BUCKET_OUTPUT', 'datos-salida')]
    )
    
    if uploaded_file is not None:
        # Guardar el archivo temporalmente
        temp_file = Path("temp_upload") / uploaded_file.name
        temp_file.parent.mkdir(exist_ok=True)
        
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Mostrar información del archivo
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Tamaño en MB
        st.write(f"**Archivo seleccionado:** {uploaded_file.name}")
        st.write(f"**Tamaño:** {file_size:.2f} MB")
        st.write(f"**Bucket de destino:** {bucket_name}")
        
        # Botón para subir el archivo
        if st.button("Subir a Backblaze B2"):
            try:
                with st.spinner("Subiendo archivo..."):
                    # Subir el archivo
                    public_url = b2_service.upload_file(
                        file_path=temp_file,
                        bucket_name=bucket_name,
                        object_name=uploaded_file.name
                    )
                    
                    # Mostrar el enlace de descarga
                    st.success("✅ ¡Archivo subido exitosamente!")
                    st.markdown(f"**Enlace de descarga:** [{public_url}]({public_url})")
                    
                    # Copiar al portapapeles
                    st.session_state["last_uploaded_url"] = public_url
                    st.button("Copiar enlace al portapapeles", 
                             on_click=lambda: st.session_state.update({"copied": True}),
                             disabled=("copied" in st.session_state and st.session_state.copied))
                    
                    if st.session_state.get("copied", False):
                        st.info("¡Enlace copiado al portapapeles!")
            
            except Exception as e:
                st.error(f"❌ Error al subir el archivo: {str(e)}")
            finally:
                # Eliminar el archivo temporal
                if temp_file.exists():
                    temp_file.unlink()

if __name__ == "__main__":
    main()
