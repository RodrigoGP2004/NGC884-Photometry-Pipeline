#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def leer_isocrona_cmd(archivo):
    with open(archivo, 'r') as f:
        lineas = f.readlines()
    header_line = None
    for line in lineas:
        # Buscamos la línea que empieza por # y tiene los nombres de columna
        if line.strip().startswith('#') and 'logAge' in line:
            header_line = line
            break 
    if header_line:
        col_names = header_line.replace('#', '').split()
        return pd.read_csv(archivo, delim_whitespace=True, comment='#', names=col_names)
    else:
        # Si falla, intentamos leer normal
        return pd.read_csv(archivo, delim_whitespace=True, comment='#')
    
def calcular_masas_por_edad(df_observado, df_isocronas, lista_edades_log, 
                            dist_mod, ebv, R_v=3.1):
    """
    Calcula las masas interpolando las isocronas dadas.
    Usa las variables globales de nombres de columnas (col_V, col_Mass, etc.)
    """
    resultados = {} # Diccionario para guardar las tablas resultantes
    
    # Buscamos qué edades reales hay disponibles en el archivo de isocronas
    edades_disponibles = df_isocronas[col_Edad].unique()

    for edad_target in lista_edades_log:
        # 1. Encontrar la edad exacta en el archivo (tu misma lógica)
        edad_real = min(edades_disponibles, key=lambda x: abs(x - edad_target))
        
        # 2. Filtrar y ordenar la isocrona
        iso_sub = df_isocronas[df_isocronas[col_Edad] == edad_real].copy()
        
        # Ordenamos por Masa (de menor a mayor) para que la interpolación funcione bien matemática
        iso_sub = iso_sub.sort_values(by=col_Mass)
        
        # 3. Desplazar la isocrona al plano observado (Magnitud Aparente V)
        # V_obs = V_teorica + (m-M) + Extincion_V
        # Extincion_V = R_v * E(B-V)
        extincion_v = R_v * ebv
        v_iso_shifted = iso_sub[col_V] + dist_mod + extincion_v
        
        masas_iso = iso_sub[col_Mass]
        
        # 4. Crear la función matemática: f(Magnitud) = Masa
        # fill_value="extrapolate" calcula masas aunque la estrella sea un poco más brillante
        # o débil que los límites de la isocrona.
        interpolador = interp1d(v_iso_shifted, masas_iso, 
                                kind='linear', 
                                bounds_error=False, 
                                fill_value="extrapolate")
        
        # 5. Aplicar a tus estrellas (df2)
        # Usamos df2['Vmag'] que es tu columna de datos observados
        masas_calculadas = interpolador(df_observado['Vmag'])
        
        # 6. Crear una tabla limpia con los resultados
        tabla_resultado = df_observado[['Vmag', 'B-V']].copy()
        tabla_resultado['Masa_Estimada'] = masas_calculadas
        tabla_resultado['Edad_Log'] = edad_real
        
        # Guardamos en el diccionario usando el log(edad) como clave
        nombre_clave = f"LogAge_{edad_real:.2f}"
        resultados[nombre_clave] = tabla_resultado
        
        print(f"-> Masas calculadas para isocrona Log(t)={edad_real:.2f}")

    return resultados

def predecir_objetos_ocultos(alpha, masas_observadas, corte_min_log_usado):
    """
    Usa la IMF calculada para predecir cuántos objetos invisibles hay.
    """
    
    # 1. DEFINIR LÍMITES (En Masas Solares)
    # Límite de quema de Hidrógeno (Estrella vs Enana Marrón)
    limite_estrella_bd = 0.08 
    # Límite de quema de Deuterio (Enana Marrón vs Planeta)
    limite_bd_planeta = 0.013
    # Límite inferior arbitrario para planetas (ej: 1 masa de Júpiter)
    limite_planeta_inf = 0.001 
    
    # 2. CALCULAR LA CONSTANTE DE NORMALIZACIÓN (k)
    # Usamos las estrellas que SÍ has visto y usado para el ajuste
    # Masa mínima que usaste para calcular la alpha (des-logaritmizada)
    masa_min_ajuste = 10**corte_min_log_usado
    # Masa máxima en tu muestra
    masa_max_ajuste = masas_observadas.max()
    
    # Contamos cuántas estrellas reales hay en ese rango fiable
    N_observado = len(masas_observadas[masas_observadas >= masa_min_ajuste])
    
    # Despejamos k de la integral: N = (k / 1-a) * [M_max^(1-a) - M_min^(1-a)]
    term_ajuste = (masa_max_ajuste**(1 - alpha)) - (masa_min_ajuste**(1 - alpha))
    k = N_observado * (1 - alpha) / term_ajuste
    
    print(f"--- Parámetros de Extrapolación ---")
    print(f"Alpha usada: {alpha:.2f}")
    print(f"Estrellas base para el cálculo: {N_observado}")
    print(f"Constante de normalización (k): {k:.2f}")
    
    # 3. FUNCIÓN PARA INTEGRAR CUALQUIER RANGO
    def integrar_imf(m_min, m_max):
        term = (m_max**(1 - alpha)) - (m_min**(1 - alpha))
        return (k / (1 - alpha)) * term

    # 4. CALCULAR POBLACIONES
    n_enanas_marrones = integrar_imf(0.013,0.08)
    n_enanas_rojas = integrar_imf(0.08,0.5)
    
    # Resultados
    print(f"\n--- PREDICCIONES TEÓRICAS ---")
    print(f"Rango Enanas Marrones ({limite_bd_planeta}-{limite_estrella_bd} M_sol):")
    print(f"-> {int(n_enanas_marrones)} enanas marrones esperadas.")
    
    print(f"\nRango Rango Enanas Rojas ({limite_planeta_inf}-{limite_bd_planeta} M_sol):")
    print(f"-> {int(n_enanas_rojas)} planetas interestelares esperados.")
    
    return n_enanas_marrones, n_enanas_rojas

def predecir_objetos_kroupa(alpha_tuya, masas_observadas, corte_min_log_usado):
    """
    Usa una IMF segmentada (tipo Kroupa) para no sobreestimar locamente 
    el número de planetas.
    """
    
    # --- 1. PUNTOS DE QUIEBRE (BREAKPOINTS) ---
    m_break_1 = 0.5   # Donde Salpeter deja de valer (0.5 masas solares)
    m_break_2 = 0.08  # Límite estelar (0.08 masas solares)
    
    # Pendientes estándar de Kroupa para masas bajas
    alpha_media = 1.3
    alpha_baja = 0.3
    
    # Límites de integración
    limite_estrella_bd = 0.08 
    limite_bd_planeta = 0.013
    limite_planeta_inf = 0.001 # 1 Júpiter
    
    # --- 2. NORMALIZACIÓN (El paso difícil) ---
    # Tenemos que asegurar que las líneas se conecten (continuidad).
    # Calculamos la constante 'k' para tu parte observada primero.
    
    masa_min_obs = 10**corte_min_log_usado
    masa_max_obs = masas_observadas.max()
    N_observado = len(masas_observadas[masas_observadas >= masa_min_obs])
    
    # K1: Constante para tu zona (Masas Altas)
    term_ajuste = (masa_max_obs**(1 - alpha_tuya)) - (masa_min_obs**(1 - alpha_tuya))
    k1 = N_observado * (1 - alpha_tuya) / term_ajuste
    
    # Ahora calculamos las constantes k2 y k3 para que las líneas se toquen
    # k1 * (0.5)^-alpha1 = k2 * (0.5)^-alpha2
    k2 = k1 * (m_break_1**(-alpha_tuya)) / (m_break_1**(-alpha_media))
    
    # k2 * (0.08)^-alpha2 = k3 * (0.08)^-alpha3
    k3 = k2 * (m_break_2**(-alpha_media)) / (m_break_2**(-alpha_baja))
    
    print(f"--- Ajuste Kroupa (Broken Power Law) ---")
    print(f"Alpha Altas (>0.5): {alpha_tuya:.2f} (Tus datos)")
    print(f"Alpha Medias (0.08-0.5): {alpha_media} (Kroupa)")
    print(f"Alpha Bajas (<0.08): {alpha_baja} (Kroupa)")
    
    # --- 3. FUNCIÓN DE INTEGRACIÓN ---
    def integrar_segmento(k, alpha, m_min, m_max):
        term = (m_max**(1 - alpha)) - (m_min**(1 - alpha))
        return (k / (1 - alpha)) * term

    # --- 4. CÁLCULOS ---
    
    # A) Enanas Marrones (0.013 a 0.08) -> Caen en la zona de alpha baja (0.3)
    # Nota: Si usas Kroupa estricto, alpha es 0.3 hasta el fondo.
    n_enanas_marrones = integrar_segmento(k3, 0.3,0.013,0.08)
    
    # B) Planetas (0.001 a 0.013) -> Alpha baja (0.3)
    n_enanas_rojas = integrar_segmento(k3, 1.3, 0.08,0.5)
    
    # Resultados
    print(f"\n--- PREDICCIONES REALISTAS ---")
    print(f"Enanas Marrones esperadas: {int(n_enanas_marrones)}")
    print(f"Enanas Rojas esperados: {int(n_enanas_rojas)}")
    
    
    return n_enanas_marrones, n_enanas_rojas

def calcular_masa_hibrida(alpha_tuya, masas_observadas, corte_min_log_usado):
    """
    Calcula la MASA TOTAL de forma HÍBRIDA:
    1. Para masas > corte: SUMA real de tus datos de Vizier.
    2. Para masas < corte: INTEGRAL teórica calibrada con tus datos.
    """
    
    # --- 1. LÍMITES FÍSICOS ---
    m_corte_usuario = 10**corte_min_log_usado  # Tu límite de confianza (ej: 3 M_sol)
    m_break_kroupa = 0.5   # Donde Kroupa cambia de pendiente
    m_lim_estelar = 0.08   # Límite Estrellas/Enanas Marrones
    m_lim_planeta = 0.013  # Límite Inferior Enanas Marrones
    
    # Pendientes Kroupa estándar
    alpha_media = 1.3 # 0.08 a 0.5
    alpha_baja = 0.3  # < 0.08
    
    # --- 2. CÁLCULO PARTE REAL (VIZIER) ---
    # Filtramos las estrellas que están por encima de tu corte
    estrellas_reales = masas_observadas[masas_observadas >= m_corte_usuario]
    
    # SUMA DIRECTA (Lo que pediste):
    masa_real_observada = np.sum(estrellas_reales)
    N_observado = len(estrellas_reales) # Necesario para calibrar la teoría
    
    print(f"--- 1. DATOS REALES (M > {m_corte_usuario:.2f} M_sol) ---")
    print(f"Estrellas contadas: {N_observado}")
    print(f"Masa Real Sumada: {masa_real_observada:.2f} M_sol")

    # --- 3. CALIBRACIÓN DE LA TEORÍA (Calcular k) ---
    # Aunque sumemos lo real, necesitamos 'k' para predecir lo invisible.
    # Usamos tus datos para encontrar la 'k' que hace que la curva teórica
    # pase justo por tus puntos.
    
    # k1 para la zona de tu alpha
    # N = (k / 1-a) * [M_max^(1-a) - M_min^(1-a)]
    masa_max_obs = masas_observadas.max()
    term_k = (masa_max_obs**(1 - alpha_tuya)) - (m_corte_usuario**(1 - alpha_tuya))
    k1 = N_observado * (1 - alpha_tuya) / term_k
    
    # k2 para la zona intermedia (0.5 - 0.5, continuidad)
    k2 = k1 * (m_break_kroupa**(-alpha_tuya)) / (m_break_kroupa**(-alpha_media))
    
    # k3 para la zona baja (< 0.08, continuidad)
    k3 = k2 * (m_lim_estelar**(-alpha_media)) / (m_lim_estelar**(-alpha_baja))

    # --- 4. INTEGRACIÓN DE LAS PARTES INVISIBLES ---
    def integrar_masa_teorica(k, alpha, m_min, m_max):
        if m_min >= m_max: return 0.0 # Por si los límites se cruzan
        exponente = 2 - alpha
        term = (m_max**exponente) - (m_min**exponente)
        return (k / exponente) * term

    # A) EL "GAP" o HUECO (Si tu corte es > 0.5)
    # Rellena teóricamente desde 0.5 hasta tu corte usando TU alpha
    masa_gap = 0
    if m_corte_usuario > m_break_kroupa:
        masa_gap = integrar_masa_teorica(k1, alpha_tuya, m_break_kroupa, m_corte_usuario)
    
    # B) ENANAS ROJAS (0.08 a 0.5) - Usando Kroupa alpha=1.3
    masa_rojas = integrar_masa_teorica(k2, alpha_media, m_lim_estelar, m_break_kroupa)
    
    # C) ENANAS MARRONES (0.013 a 0.08) - Usando Kroupa alpha=0.3
    masa_bd = integrar_masa_teorica(k3, alpha_baja, m_lim_planeta, m_lim_estelar)

    # --- 5. RESULTADOS FINALES ---
    masa_total = masa_real_observada + masa_gap + masa_rojas + masa_bd
    
    print(f"\n--- 2. ESTIMACIONES TEÓRICAS (Extrapolación) ---")
    if masa_gap > 0:
        print(f"Masa en el 'Hueco' (0.5 - {m_corte_usuario:.2f}): {masa_gap:.2f} M_sol")
    print(f"Masa Enanas Rojas (0.08 - 0.5):   {masa_rojas:.2f} M_sol")
    print(f"Masa Enanas Marrones (< 0.08):    {masa_bd:.2f} M_sol")
    
    print(f"\n" + "="*40)
    print(f"MASA TOTAL HÍBRIDA: {masa_total:.2f} M_sol")
    print("="*40)
    
    return masa_total

def predecir_masivas_faltantes(alpha, masas_observadas, corte_min_log_usado, tope_teorico=150):
    """
    Calcula cuántas estrellas más masivas que la mayor observada 
    deberían haber existido según la IMF.
    
    Parámetros:
    - tope_teorico: Límite máximo de masa estelar (aprox 120-150 M_sol).
    """
    
    # 1. ENCONTRAR LA MASA MÁXIMA REAL QUE HAS VISTO
    masa_max_obs = masas_observadas.max()
    
    # 2. CALCULAR K (NORMALIZACIÓN) - IGUAL QUE ANTES
    # Usamos el rango fiable donde hiciste el ajuste lineal
    masa_min_ajuste = 10**corte_min_log_usado
    
    # Filtramos las estrellas que usaste para el ajuste
    masas_validas = masas_observadas[masas_observadas >= masa_min_ajuste]
    N_observado = len(masas_validas)
    
    # Fórmula de k (basada en la integral definida del rango observado)
    term_ajuste = (masa_max_obs**(1 - alpha)) - (masa_min_ajuste**(1 - alpha))
    k = N_observado * (1 - alpha) / term_ajuste
    
    print(f"--- Datos del Ajuste ---")
    print(f"Estrella más masiva observada: {masa_max_obs:.2f} M_sol")
    print(f"Tope teórico asumido: {tope_teorico} M_sol")
    print(f"Constante k: {k:.2f}")

    # 3. INTEGRAR HACIA ARRIBA (De Max_Observada a Tope_Teorico)
    def integrar_imf(m_min, m_max):
        # Evitamos errores si la alfa es exactamente 1 (muy raro, pero posible)
        if alpha == 1:
            return k * (np.log(m_max) - np.log(m_min))
        else:
            term = (m_max**(1 - alpha)) - (m_min**(1 - alpha))
            return (k / (1 - alpha)) * term

    # Calculamos
    n_masivas_teoricas = integrar_imf(25, tope_teorico)
    
    # 4. RESULTADOS E INTERPRETACIÓN
    print(f"\n--- PREDICCIÓN DE GIGANTES ---")
    print(f"En el rango {masa_max_obs:.1f} - {tope_teorico} M_sol:")
    print(f"-> Predicción matemática: {n_masivas_teoricas:.4f} estrellas.")
    
    # Interpretación física para el informe
    if n_masivas_teoricas < 1:
        print("\nInterpretación: Es probable que nunca se formara ninguna estrella tan masiva.")
    else:
        print(f"\nInterpretación: Según la estadística, deberían haberse formado {int(round(n_masivas_teoricas))} estrellas más.")
        print("Dado que no están, las opciones son:")
        print("1. Ya explotaron como Supernovas (si su vida < 14 Myr).")
        print("2. Fueron expulsadas del cúmulo.")
        print("3. La estadística de 'números pequeños' hizo que no se formaran por azar.")

    return n_masivas_teoricas

# --- EJEMPLO DE USO ---
# predecir_masivas_faltantes(alpha_calculada, masas, corte_min_log, tope_teorico=150)

# --- EJEMPLO DE USO ---
# Asegúrate de pasarle tu alpha calculada, tus masas y tu log(corte)
# calcular_masa_hibrida(alpha_calculada, masas, corte_min_log)

# EJECUTAR LA NUEVA FUNCIÓN

#%%

#Cargar tablas
_ColorInd = np.load('/home/rodrigogp/Documents/TEA/SavedArrays/ColorInd.npy')
_V = np.load('/home/rodrigogp/Documents/TEA/SavedArrays/V.npy')

#Cargo los datos de Vizier
df = pd.read_csv('/home/rodrigogp/Documents/TEA/NGC884.txt', sep='\s+')
df2 = df[df['M'] == 'chi']
color = df2['B-V']
mag = df2['Vmag']

#Cargo las isocronas
iso = leer_isocrona_cmd('/home/rodrigogp/Documents/TEA/isochrones.dat')
col_B = 'Bmag'
col_V = 'Vmag'
col_Edad = 'logAge'
col_Mass = 'Mini'
reddening = 0.53       # E(B-V)  (Mueve der/izq)
dist = 11.85 # m-M
dist_pc = 10 ** ((dist + 5)/5)
dist_ly = dist_pc * 3.26156
R_v = 3.1

#Seleccionamos tres isocronas
edades = [7.05, 7.09, 7.13]
colores = ['blue','red','green']
edades_en_archivo = iso[col_Edad].unique()

plt.plot(_ColorInd,_V,'o',color='orange',label='Nuestros datos')
plt.plot(color,mag,'o', color='black',label='Datos de Vizier')

for i, edad_target in enumerate(edades):
    # Encontrar la edad disponible más cercana a la que queremos
    edad_real = min(edades_en_archivo, key=lambda x: abs(x - edad_target))
    
    # --- AQUÍ ESTÁ LA MAGIA: FILTRADO ---
    # Creamos un sub-dataframe SOLO con esa edad
    iso_sub = iso[iso[col_Edad] == edad_real].copy()
    #iso_sub = iso_sub.sort_values(by=col_B) # Ordenar para que la línea salga bien
    iso_sub = iso_sub.sort_values(by=col_Mass)
    
    # Cálculos
    iso_x = (iso_sub[col_B] - iso_sub[col_V]) + reddening
    iso_y = iso_sub[col_V] + (R_v * reddening) + dist
    
    # Pintar
    millones_anyos = (10**edad_real) / 1e6
    plt.plot(iso_x, iso_y, '-', color=colores[i], linewidth=2,label=f'Log(t)={edad_real:.2f} (~{millones_anyos:.1f} Myr)')

#Graficar
plt.gca().invert_yaxis()
plt.legend(loc='lower right', fontsize=8)
plt.grid()
plt.xlabel("(B-V)")
plt.ylabel("V")
plt.ylim(18,5)
#plt.xlim(0,1)
plt.rcParams['figure.dpi'] = 400
plt.show()

#%%
tablas_de_masas = calcular_masas_por_edad(
    df_observado=df2,       # Tus datos de Vizier filtrados
    df_isocronas=iso,       # Tu archivo de isocronas cargado
    lista_edades_log=edades,# Tu lista [7.05, 7.09, 7.13]
    dist_mod=dist,          # Tu variable 'dist' (11.85)
    ebv=reddening,          # Tu variable 'reddening' (0.53)
    R_v=R_v                 # Tu variable R_v (3.1)
)

#%%
key2 = list(tablas_de_masas.keys())[1] 
Masas2 = tablas_de_masas[key2]
masa_max = Masas2['Masa_Estimada'].max()
masa_min = Masas2['Masa_Estimada'].min()

#%%
# =============================================================================
# GRAFICAR HISTOGRAMA DE MASAS
# =============================================================================
df_plot = tablas_de_masas[key2]
plt.figure(figsize=(8, 6))

# Usamos los mismos datos (df_plot) que elegimos arriba
masas = df_plot['Masa_Estimada']
bins_log = np.geomspace(masas.min(), masas.max(),21)

# bins='auto' deja que numpy decida el mejor número de barras
# o puedes poner un número fijo, ej: bins=15
plt.hist(masas, bins=bins_log, color='skyblue', edgecolor='black', alpha=0.7)

plt.xlabel('Masa Solar ($M_{\odot}$)', fontsize=12)
plt.ylabel('Número de Estrellas', fontsize=12)
#plt.title(f'Histograma de Masas (12.3 Myr)', fontsize=14)
plt.grid(axis='y', alpha=0.5)

plt.show()

#%%
y_counts, edges = np.histogram(masas, bins=bins_log)
x_centers = (edges[:-1] + edges[1:]) / 2
bin_widths = np.diff(edges) #dM
densidad = y_counts / bin_widths #dN/dM

# --- 3. PASAR A LOGARITMOS (Log-Log) ---
# Filtramos bins vacíos (donde densidad es 0) porque log(0) da error
mask = densidad > 0
log_m = np.log10(x_centers[mask])
log_xi = np.log10(densidad[mask])

# --- 4. AJUSTE LINEAL (Calcular la pendiente) ---
# OJO: La ley de potencias solo vale para estrellas masivas.
# Las estrellas pequeñas (poca masa) suelen faltar porque son muy débiles para verse.
# Tienes que definir un "Corte de Masa" visualmente.
# Para NGC 884, prueba cortando en log(m) > 0.5 (aprox 3 masas solares)
corte_min_log = 0.217  # Ajusta este número mirando tu gráfica

mask_fit = log_m >= corte_min_log
x_fit = log_m[mask_fit]
y_fit = log_xi[mask_fit]

# Ajuste polinómico de grado 1 (Recta: y = mx + c)
if len(x_fit) > 1:
    pendiente, intercepto = np.polyfit(x_fit, y_fit, 1)
    alpha_calculada = -pendiente # La alpha se define positiva, la pendiente es negativa
else:
    pendiente, intercepto = 0, 0
    alpha_calculada = 0
    print("¡Cuidado! No hay suficientes puntos para el ajuste. Baja el 'corte_min_log'.")

# --- 5. GRAFICAR LA IMF ---
plt.figure(figsize=(8, 6))

# Puntos observados
plt.scatter(log_m, log_xi, color='black', label='Datos (dN/dM)')

# Línea de ajuste (Tu resultado)
y_teorico = pendiente * x_fit + intercepto
plt.plot(x_fit, y_teorico, 'r-', linewidth=2, 
         label=f'Ajuste: $\\alpha = {alpha_calculada:.2f}$')

# Línea de Salpeter (Referencia teórica: pendiente -2.35)
# La pintamos pasando por el punto medio de tu ajuste para comparar inclinación
x_mid = np.mean(x_fit)
y_mid = np.mean(y_fit)
y_salpeter = -2.35 * (x_fit - x_mid) + y_mid
plt.plot(x_fit, y_salpeter, 'b--', alpha=0.6, label='Salpeter ($\\alpha=2.35$)')

plt.xlabel('Log10 (Masa / $M_{\odot}$)', fontsize=12)
plt.ylabel('Log10 (Densidad $\\xi$)', fontsize=12)
#plt.title(f'Función Inicial de Masas (IMF) - NGC 884', fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.show()

#%%
predecir_objetos_ocultos(alpha_calculada, masas, corte_min_log)
#%%
predecir_objetos_kroupa(alpha_calculada, masas, corte_min_log)

#%%
calcular_masa_hibrida(alpha_calculada, masas, corte_min_log)

#%%
predecir_masivas_faltantes(alpha_calculada, masas, corte_min_log, tope_teorico=150)