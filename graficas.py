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
    
    # Limitar N para la generación de colores si es excesivamente grande
    N_colors = min(n, 20) 
    
    # Crea el mapa de colores: de claro a oscuro (Index 0 es claro, Index N_colors-1 es oscuro)
    cmap_seq = mcolors.LinearSegmentedColormap.from_list(
        "custom_ramp_no_white", [LIGHT_COLORED_ANCHOR, DARK_ANCHOR], N=N_colors
    )
    # Genera la lista de colores
    palette = [mcolors.to_hex(cmap_seq(i / (N_colors - 1.0))) for i in range(N_colors)]
    
    # Si n > 20, repite los colores más claros para las categorías menos frecuentes
    if n > 20:
        palette.extend([palette[0]] * (n - 20)) 
    
    return palette

# Función para crear un gradiente horizontal en una barra
def plot_gradient_barh(ax, y, width, bar_height, color_start, color_end):
    # Genera un gradiente lineal del color inicial al color final
    cmap = mcolors.LinearSegmentedColormap.from_list('gradient', [color_start, color_end], N=256)
    # Define la resolución para el gradiente
    X = np.linspace(0, 1, 256).reshape(1, -1)
    height_res = 20
    X = np.tile(X, (height_res, 1))
    # Dibuja la imagen del gradiente sobre la barra
    ax.imshow(X, extent=[0, width, y - bar_height/2, y + bar_height/2], aspect='auto', cmap=cmap, zorder=2)

# --- 2. ENTRADA DE DATOS (File Uploader) ---
st.title("📊 Herramienta de Visualización de Incidentes Urbanos")

# Opción para subir el archivo
uploaded_file = st.file_uploader(
    "📤 Sube tu archivo CSV o XLSX para comenzar el análisis:", 
    type=["csv", "xlsx"],
    help="El archivo debe contener las columnas de 'colonia' y 'tipo_de_reporte_(incidente)'."
)

df = None
if uploaded_file is not None:
    # Carga el archivo según su tipo
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Se asume la primera hoja para archivos XLSX
            df = pd.read_excel(uploaded_file)
        
        st.success("✅ Archivo cargado correctamente. Usa el menú lateral para configurar el gráfico.")
    
    except Exception as e:
        st.error(f"❌ Error al leer el archivo. Asegúrate de que el formato es correcto. Detalles: {e}")

# --- 3. FLUJO PRINCIPAL DE STREAMLIT ---
if df is not None:

    st.sidebar.header("⚙️ Configuración del Análisis")
    
    # 3.1 Tipo de análisis
    data_type = st.sidebar.radio(
        "Selecciona el Tipo de Análisis:",
        ('Top Incidentes', 'Top Colonias (General)', 'Top Colonias (Lluvias)')
    )

    # 3.2 Slider para Top N
    # Asegurarse de que el DataFrame filtrado no esté vacío antes de calcular max_n
    try:
        source_df_check = df[df['?el_reporte_referente_a_las_lluvias?'] == 'si'] if data_type == 'Top Colonias (Lluvias)' else df
        
        if data_type == 'Top Incidentes':
            unique_count = len(source_df_check['tipo_de_reporte_(incidente)'].dropna().unique())
        else:
            unique_count = len(source_df_check['colonia'].dropna().unique())
            
        max_n = min(unique_count, 20) # Limitar a un máximo de 20 para buena visualización
        
    except KeyError:
        st.error("❌ Error: Verifica que tu archivo contenga las columnas 'colonia' y 'tipo_de_reporte_(incidente)'.")
        st.stop()
        
    if max_n < 2:
         st.warning("No hay suficientes datos únicos para generar un Top 2. Filtra tu archivo.")
         st.stop()
        
    top_n = st.sidebar.slider(
        "Selecciona el número Top N:",
        min_value=2, max_value=max_n, value=min(10, max_n)
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
    
    col_name = 'tipo_de_reporte_(incidente)' if data_type == 'Top Incidentes' else 'colonia'
    is_lluvias = data_type == 'Top Colonias (Lluvias)'

    if is_lluvias:
        source_df = df[df['?el_reporte_referente_a_las_lluvias?'] == 'si'].copy()
        if source_df.empty:
             st.warning("⚠️ El filtro de lluvias no arrojó datos. Intenta con 'Top Colonias (General)'.")
             st.stop()
    else:
        source_df = df.copy()

    # Obtener el Top N
    data = source_df[col_name].value_counts().nlargest(top_n).reset_index()
    data.columns = ['etiqueta', 'cantidad']
    
    # Etiquetas de ejes y título por defecto
    ylabel = 'Colonia' if col_name == 'colonia' else 'Incidente'
    default_title = f'Top {top_n} {ylabel}s'

    # Ordenar datos para barras (ascendente para horizontal, descendente para vertical)
    if chart_type == 'Barras':
        data = data.sort_values('cantidad', ascending=(orientation == 'horizontal'))

    # --- 5. TÍTULO DINÁMICO ---
    custom_title = st.text_input("📝 Editar Título del Gráfico:", value=default_title)


    # --- 6. FUNCIÓN DE PLOTEO PRINCIPAL ---
    
    # Generar paleta de N colores (la paleta es secuencial, oscura a clara)
    palette = generate_sequential_palette(top_n)
    fig, ax = plt.subplots(figsize=(12, 7))

    if chart_type == 'Barras':
        
        if orientation == 'horizontal':
            # Horizontal con gradiente personalizado
            bar_height_val = 0.6
            for i, (cantidad, etiqueta) in enumerate(zip(data['cantidad'], data['etiqueta'])):
                
                # Invertir el color si es necesario para que el más intenso esté arriba (mayor cantidad)
                # Como 'data' está sorted ascending, el index 'i' es nuestro rango de 0 (bottom) a N (top)
                color_start = palette[i]

                # Generar el color final del gradiente (aclarado para contraste)
                rgb = mcolors.hex2color(color_start)
                lighter_color = mcolors.rgb2hex([min(1, c + 0.25) for c in rgb]) 
                
                plot_gradient_barh(ax, i, cantidad, bar_height_val, color_start, lighter_color)

            ax.set_yticks(range(top_n))
            ax.set_yticklabels(data['etiqueta'])
            ax.set_xlabel('Cantidad de Reportes')
            ax.set_ylabel("") 
            ax.set_xlim(0, data['cantidad'].max() * 1.15)
            ax.tick_params(axis='y', which='both', length=0)
            ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)

        else: # orientation == 'vertical'
            # Vertical (usando solo el color base secuencial)
            ax.bar(data['etiqueta'], data['cantidad'], color=palette, zorder=3)
            ax.set_xlabel(ylabel) 
            ax.set_ylabel('Cantidad de Reportes')
            plt.xticks(rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
            ax.set_ylim(0, data['cantidad'].max() * 1.15)
            ax.set_xticklabels(data['etiqueta'], ha='right')


    elif chart_type in ['Pastel', 'Dona']:
        
        # Para Pasteles/Donas: Agrupar el resto en 'Otros' si es necesario
        total_sum = source_df[col_name].value_counts().sum()
        top_n_sum = data['cantidad'].sum()
        other_sum = total_sum - top_n_sum
        
        plot_data = data.copy()
        plot_palette = palette.copy()
        
        if other_sum > 0:
            plot_data.loc[len(plot_data)] = ['Otros', other_sum]
            # Regenerar/Extender la paleta para incluir el color de 'Otros' (tono claro)
            plot_palette.append(LIGHT_COLORED_ANCHOR) 
        
        # Plotear Pasteles/Donas
        ax.pie(plot_data['cantidad'], labels=plot_data['etiqueta'], autopct='%1.1f%%', colors=plot_palette, startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
        ax.axis('equal') 
        
        if chart_type == 'Dona':
            # Crear un círculo blanco central para el efecto de dona
            centre_circle = plt.Circle((0,0), 0.65, color='white', fc='white',linewidth=1.25, edgecolor='gray')
            ax.add_artist(centre_circle)
    
    # --- 7. RENDERIZADO FINAL ---
    ax.set_title(custom_title, fontsize=16)
    st.pyplot(fig)


# --- REQUISITOS DEL CÓDIGO ---
st.markdown("---")
st.markdown("### 🛠️ Requisitos del Código (`requirements.txt`)")
st.code("streamlit\npandas\nmatplotlib\nnumpy")
