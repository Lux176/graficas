import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import io

# --- 1. CONFIGURACIÓN DE DATOS Y ESTILOS ---

# Definición de la paleta de colores secuencial acordada (Ancla Oscura y Ancla Clara no-blanca)
DARK_ANCHOR = '#853815'
LIGHT_COLORED_ANCHOR = '#e28f54'

# Función para generar la paleta secuencial de N colores (de más oscuro a más claro)
def generate_sequential_palette(n):
    if n <= 1:
        return [DARK_ANCHOR]
    
    # Crea el mapa de colores: de claro a oscuro (Index 0 es claro, Index n-1 es oscuro)
    cmap_seq = mcolors.LinearSegmentedColormap.from_list(
        "custom_ramp_no_white", [LIGHT_COLORED_ANCHOR, DARK_ANCHOR], N=n
    )
    # Muestra los colores de más claro (bottom) a más oscuro (top).
    return [mcolors.to_hex(cmap_seq(i / (n - 1.0))) for i in range(n)]

# Función para crear un gradiente horizontal en una barra
def plot_gradient_barh(ax, y, width, bar_height, color_start, color_end):
    cmap = mcolors.LinearSegmentedColormap.from_list('gradient', [color_start, color_end], N=256)
    X = np.linspace(0, 1, 256).reshape(1, -1)
    height_res = 20
    X = np.tile(X, (height_res, 1))
    ax.imshow(X, extent=[0, width, y - bar_height/2, y + bar_height/2], aspect='auto', cmap=cmap, zorder=2)

# --- 2. CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        # ASUMIENDO que el archivo CSV está en la misma carpeta
        df = pd.read_csv('opc (2).xlsx - Datos_Procesados.csv')
        return df
    except FileNotFoundError:
        st.error("Error: El archivo 'opc (2).xlsx - Datos_Procesados.csv' no fue encontrado. Colócalo en la misma carpeta que 'app.py'.")
        return None

df = load_data()

if df is not None:

    # --- 3. CONTROLES DEL SIDEBAR ---
    st.sidebar.header("⚙️ Configuración del Análisis")
    
    # 3.1 Tipo de análisis (Incidentes generales o Colonias)
    data_type = st.sidebar.radio(
        "Selecciona el Tipo de Análisis:",
        ('Top Incidentes', 'Top Colonias (General)', 'Top Colonias (Lluvias)')
    )

    # 3.2 Slider para Top N
    max_n = len(df['colonia'].unique()) if data_type != 'Top Incidentes' else len(df['tipo_de_reporte_(incidente)'].unique())
    max_n = min(max_n, 20) # Limitar a un máximo de 20 para visualización
    top_n = st.sidebar.slider(
        "Selecciona el número Top N:",
        min_value=2, max_value=max_n, value=10
    )

    # 3.3 Tipo de Gráfico
    chart_type = st.sidebar.selectbox(
        "Selecciona el Tipo de Gráfico:",
        ('Barras', 'Pastel', 'Dona')
    )

    # 3.4 Orientación (Solo para Barras)
    orientation = 'horizontal'
    if chart_type == 'Barras':
        orientation = st.sidebar.radio(
            "Orientación de la Barra:",
            ('Horizontal', 'Vertical')
        ).lower()

    # --- 4. PREPARACIÓN DE DATOS DINÁMICA ---
    
    # Definir la columna a analizar y etiquetas
    col_name = 'tipo_de_reporte_(incidente)' if data_type == 'Top Incidentes' else 'colonia'
    is_lluvias = data_type == 'Top Colonias (Lluvias)'

    if is_lluvias:
        source_df = df[df['?el_reporte_referente_a_las_lluvias?'] == 'si'].copy()
    else:
        source_df = df.copy()

    # Obtener el Top N
    data = source_df[col_name].value_counts().nlargest(top_n).reset_index()
    data.columns = ['etiqueta', 'cantidad']
    
    # Etiquetas de ejes y título por defecto
    xlabel = 'Cantidad de Reportes'
    ylabel = 'Colonia' if col_name == 'colonia' else 'Tipo de Incidente'
    default_title = f'Top {top_n} {ylabel}s'

    # Ordenar datos para barras (ascendente para horizontal, descendente para vertical)
    if chart_type == 'Barras':
        data = data.sort_values('cantidad', ascending=(orientation == 'horizontal'))


    # --- 5. TÍTULO DINÁMICO ---
    custom_title = st.text_input("📝 Editar Título del Gráfico:", value=default_title)


    # --- 6. FUNCIÓN DE PLOTEO PRINCIPAL ---
    
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = generate_sequential_palette(top_n)

    if chart_type == 'Barras':
        
        if orientation == 'horizontal':
            # Horizontal con gradiente
            bar_height_val = 0.6
            for i, (cantidad, etiqueta) in enumerate(zip(data['cantidad'], data['etiqueta'])):
                color_start = palette[i]
                # Generar el color final del gradiente (aclarado para contraste)
                rgb = mcolors.hex2color(color_start)
                lighter_color = mcolors.rgb2hex([min(1, c + 0.25) for c in rgb]) 
                
                plot_gradient_barh(ax, i, cantidad, bar_height_val, color_start, lighter_color)

            ax.set_yticks(range(top_n))
            ax.set_yticklabels(data['etiqueta'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("") # Quitamos el ylabel para no saturar el espacio
            ax.set_xlim(0, data['cantidad'].max() * 1.15)
            ax.tick_params(axis='y', which='both', length=0)
            ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)

        else: # orientation == 'vertical'
            # Vertical (usando solo el color base secuencial)
            ax.bar(data['etiqueta'], data['cantidad'], color=palette, zorder=3)
            ax.set_xlabel(ylabel) 
            ax.set_ylabel(xlabel)
            plt.xticks(rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
            ax.set_ylim(0, data['cantidad'].max() * 1.15)


    elif chart_type in ['Pastel', 'Dona']:
        
        # Para Pastel/Dona: Agrupar el resto en 'Otros' si es necesario
        if len(data) < len(source_df[col_name].unique()):
            total_sum = source_df[col_name].value_counts().sum()
            top_n_sum = data['cantidad'].sum()
            other_sum = total_sum - top_n_sum
            
            if other_sum > 0:
                data.loc[len(data)] = ['Otros', other_sum]
                # Regenerar la paleta para incluir el color de 'Otros'
                palette = generate_sequential_palette(top_n + 1)
                
        
        # Plotear Pasteles/Donas
        ax.pie(data['cantidad'], labels=data['etiqueta'], autopct='%1.1f%%', colors=palette, startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        ax.axis('equal') 
        
        if chart_type == 'Dona':
            # Crear un círculo blanco central para el efecto de dona
            centre_circle = plt.Circle((0,0), 0.65, color='white', fc='white',linewidth=1.25, edgecolor='black')
            ax.add_artist(centre_circle)
    
    # --- 7. RENDERIZADO FINAL ---
    ax.set_title(custom_title, fontsize=16)
    st.pyplot(fig)

    st.markdown("---")
    st.write(f"📊 La gráfica se actualiza en tiempo real mostrando el Top {top_n} {ylabel}s.")


# --- REQUISITOS DEL CÓDIGO ---
st.markdown("### 🛠️ Requisitos del Código (`requirements.txt`)")
st.code("streamlit\npandas\nmatplotlib\nnumpy")
