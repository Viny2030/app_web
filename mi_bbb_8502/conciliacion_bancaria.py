import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from conciliacion import main_generar_y_procesar

# Configuración de la página
st.set_page_config(
    page_title="Auditoría de Conciliación", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO PARA VIBE MODERNO ---
st.markdown("""
<style>
    /* Gradientes y sombras modernas */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    /* Animaciones sutiles */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    /* Estilo para tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    
    /* Badge style */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER MODERNO ---
st.markdown("""
<div class="main-header fade-in">
    <h1 style="color: white; margin: 0; text-align: center; font-size: 2.5rem;">
        🔍 Sistema de Auditoría y Conciliación Bancaria
    </h1>
    <p style="color: rgba(255,255,255,0.9); text-align: center; margin-top: 0.5rem; font-size: 1.1rem;">
        Powered by <strong>Machine Learning</strong> • Detección Inteligente de Anomalías
    </p>
</div>
""", unsafe_allow_html=True)


# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Verificamos si existe el archivo CSV, si no, lo generamos
    if not os.path.exists('informe_conciliacion.csv'):
        with st.spinner('Generando datos de conciliación...'):
            main_generar_y_procesar()
    
    # Cargamos el archivo CSV
    try:
        df = pd.read_csv('informe_conciliacion.csv')
        # Convertimos fechas
        df['fecha_banco'] = pd.to_datetime(df['fecha_banco'], errors='coerce')
        df['fecha_contable'] = pd.to_datetime(df['fecha_contable'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Error: No se encuentra el archivo 'informe_conciliacion.csv'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame()


df = load_data()

if not df.empty:
    # --- BARRA LATERAL ---
    st.sidebar.header("Filtros")
    estado_filtro = st.sidebar.multiselect(
        "Filtrar por Estado:",
        options=df['clasificación_auditoría'].unique(),
        default=df['clasificación_auditoría'].unique()
    )

    df_filtered = df[df['clasificación_auditoría'].isin(estado_filtro)]

    # --- KPIS PRINCIPALES CON ESTILO MODERNO ---
    st.markdown("### 📈 Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transacciones",
            value=f"{len(df_filtered):,}",
            delta=None,
            delta_color="normal"
        )
    
    diferencia_total = df_filtered['diferencia_monto'].sum()
    with col2:
        delta_val = abs(diferencia_total) / 1000 if diferencia_total != 0 else 0
        st.metric(
            label="Diferencia Total",
            value=f"${diferencia_total:,.2f}",
            delta=f"${delta_val:,.0f}K" if delta_val > 0 else None,
            delta_color="inverse"
        )

    no_conciliados = df_filtered[df_filtered['conciliado'] == False].shape[0]
    porcentaje_pendiente = (no_conciliados / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    with col3:
        st.metric(
            label="Pendientes de Revisión",
            value=f"{no_conciliados:,}",
            delta=f"{porcentaje_pendiente:.1f}%",
            delta_color="inverse"
        )

    promedio_dias = df_filtered['diferencia_dias'].mean()
    with col4:
        st.metric(
            label="Promedio Desfase",
            value=f"{promedio_dias:.1f} días",
            delta=None
        )

    st.markdown("---")

    # --- PESTAÑAS DE ANÁLISIS ---
    tab1, tab2, tab3 = st.tabs(["📊 Resumen Visual", "🤖 Detección de Fraude (IA)", "📋 Datos Detallados"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Distribución de Clasificaciones")
            # Gráfico de pastel moderno con colores personalizados
            colors_pie = px.colors.qualitative.Set3
            fig_pie = px.pie(
                df_filtered, 
                names='clasificación_auditoría', 
                title='Estado de la Conciliación',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4  # Donut chart más moderno
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig_pie.update_layout(
                font=dict(size=12),
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("💰 Distribución de Diferencias Monetarias")
            # Histograma con gradiente y mejor diseño
            fig_hist = px.histogram(
                df_filtered, 
                x='diferencia_monto', 
                title='Histograma de Diferencias ($)',
                nbins=30,
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_traces(
                marker=dict(line=dict(color='white', width=1)),
                opacity=0.8
            )
            fig_hist.update_layout(
                xaxis_title="Diferencia en Dólares ($)",
                yaxis_title="Frecuencia",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Gráfico adicional de línea temporal
        st.markdown("---")
        st.subheader("📅 Evolución Temporal de Diferencias")
        if 'fecha_banco' in df_filtered.columns:
            df_temporal = df_filtered.groupby(df_filtered['fecha_banco'].dt.date).agg({
                'diferencia_monto': 'sum',
                'conciliado': 'count'
            }).reset_index()
            df_temporal.columns = ['Fecha', 'Diferencia Total', 'Cantidad']
            
            fig_timeline = px.line(
                df_temporal,
                x='Fecha',
                y='Diferencia Total',
                title='Tendencia de Diferencias en el Tiempo',
                markers=True,
                color_discrete_sequence=['#f5576c']
            )
            fig_timeline.update_traces(line=dict(width=3), marker=dict(size=8))
            fig_timeline.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    with tab2:
        st.subheader("🤖 Detección Inteligente de Anomalías con IA")
        
        # Info box moderno
        st.info("""
        🧠 **Algoritmo Isolation Forest**: Utiliza técnicas de Machine Learning para identificar 
        patrones anómalos en las transacciones basándose en diferencias de monto y días.
        """)
        
        # Slider para ajustar sensibilidad
        contamination_level = st.slider(
            "🔧 Nivel de Sensibilidad de Detección",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Ajusta qué porcentaje de transacciones se consideran anómalas"
        )

        # PREPARACIÓN PARA ML
        features = ['diferencia_monto', 'diferencia_dias']
        X = df_filtered[features].fillna(0)

        if len(X) > 0:
            # ALGORITMO 1: ISOLATION FOREST
            with st.spinner('🔍 Analizando transacciones con IA...'):
                model_iso = IsolationForest(contamination=contamination_level, random_state=42)
                df_filtered['anomalia_score'] = model_iso.fit_predict(X)
                # -1 es anomalía, 1 es normal
                anomalias = df_filtered[df_filtered['anomalia_score'] == -1]

            # Alertas visuales modernas
            if len(anomalias) > 0:
                st.error(f"""
                ⚠️ **ALERTA**: Se han detectado **{len(anomalias)} transacciones anómalas** 
                ({len(anomalias)/len(df_filtered)*100:.1f}% del total) que requieren atención inmediata.
                """)
            else:
                st.success("✅ No se detectaron anomalías con el nivel de sensibilidad actual.")

            # Visualización de Anomalías mejorada
            fig_scatter = px.scatter(
                df_filtered,
                x='diferencia_dias',
                y='diferencia_monto',
                color=df_filtered['anomalia_score'].map({1: '✅ Normal', -1: '🚨 Anomalía'}),
                color_discrete_map={'✅ Normal': '#667eea', '🚨 Anomalía': '#f5576c'},
                hover_data=['concepto_banco', 'fecha_banco', 'monto_banco'],
                title="🗺️ Mapa de Anomalías: Monto vs Días de Diferencia",
                labels={
                    'diferencia_dias': 'Diferencia en Días',
                    'diferencia_monto': 'Diferencia en Monto ($)'
                },
                size_max=15
            )
            fig_scatter.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            if len(anomalias) > 0:
                st.markdown("#### 🚨 Listado de Anomalías Detectadas")
                # Tabla mejorada con formato
                anomalias_display = anomalias[[
                    'fecha_banco', 'concepto_banco', 'monto_banco', 
                    'diferencia_monto', 'diferencia_dias'
                ]].copy()
                anomalias_display['diferencia_monto'] = anomalias_display['diferencia_monto'].apply(
                    lambda x: f"${x:,.2f}"
                )
                st.dataframe(
                    anomalias_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Botón de descarga
                csv_anomalias = anomalias_display.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Anomalías como CSV",
                    data=csv_anomalias,
                    file_name=f"anomalias_detectadas_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    with tab3:
        st.subheader("📋 Explorador de Datos Completo")
        
        # Filtros adicionales
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_monto = st.number_input(
                "💰 Monto Mínimo ($)",
                min_value=float(df_filtered['diferencia_monto'].min()) if len(df_filtered) > 0 else 0.0,
                max_value=float(df_filtered['diferencia_monto'].max()) if len(df_filtered) > 0 else 10000.0,
                value=0.0
            )
        with col_f2:
            max_dias = st.number_input(
                "📅 Días Máximos de Diferencia",
                min_value=0,
                max_value=int(df_filtered['diferencia_dias'].max()) if len(df_filtered) > 0 else 30,
                value=int(df_filtered['diferencia_dias'].max()) if len(df_filtered) > 0 else 30
            )
        
        # Aplicar filtros adicionales
        df_display = df_filtered[
            (df_filtered['diferencia_monto'] >= min_monto) &
            (df_filtered['diferencia_dias'] <= max_dias)
        ]
        
        st.markdown(f"**Mostrando {len(df_display)} de {len(df_filtered)} transacciones**")
        
        # Tabla con estilo mejorado
        def highlight_conciliado(val):
            if val == False:
                return 'background-color: #ff6b6b; color: white'
            else:
                return 'background-color: #51cf66; color: white'
        
        styled_df = df_display.style.applymap(
            highlight_conciliado,
            subset=['conciliado']
        ).format({
            'diferencia_monto': '${:,.2f}',
            'monto_banco': '${:,.2f}',
            'monto_contable': '${:,.2f}',
            'diferencia_dias': '{:.1f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Estadísticas rápidas
        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Conciliados", f"{df_display['conciliado'].sum():,}")
        with col_s2:
            st.metric("No Conciliados", f"{(~df_display['conciliado']).sum():,}")
        with col_s3:
            tasa_conciliacion = (df_display['conciliado'].sum() / len(df_display) * 100) if len(df_display) > 0 else 0
            st.metric("Tasa de Conciliación", f"{tasa_conciliacion:.1f}%")

else:
    st.warning("⏳ Esperando archivo de datos...")
    st.info("💡 La aplicación generará automáticamente los datos cuando esté lista.")