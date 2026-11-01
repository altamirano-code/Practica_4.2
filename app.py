"""
Práctica 4.2 - Regresión Lineal y Correlación
Análisis y Exploración de Datos (TecNM INC-2505) - Unidad 4
Caso: Predicción del Consumo Per Cápita de Agua Potable (CPC)

App de Streamlit basada en el código proporcionado por el profesor
(practica4_2_corregido_graficas.py).
"""
import io
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Práctica 4.2 - Consumo de agua",
    page_icon="💧",
    layout="wide",
)
sns.set_theme(style="whitegrid")

VARIABLES_X = ["temperatura_max", "humedad_max", "personas_familia", "consumo_bidon_pc"]
VARIABLE_Y = "cpc_consumo"

SQL_SIMULACION = """
CREATE TABLE telemetria_agua AS
WITH base AS (
    SELECT
        residencia_id,
        ROUND(18 + (random() * 14), 2) AS temperatura_max,       -- 18°C a 32°C
        ROUND(50 + (random() * 40), 2) AS humedad_max,           -- 50% a 90%
        ROUND(3 + (random() * 4), 0)   AS personas_familia,      -- 3 a 7 hab/vivienda
        ROUND(0.2 + (random() * 1.8), 2) AS consumo_bidon_pc,    -- 0.2 a 2.0 lt/hab-día
        (random() * 6.0 - 3.0) AS ruido                          -- error aleatorio de -3 a 3
    FROM generate_series(1, {n}) AS t(residencia_id)
)
SELECT
    residencia_id, temperatura_max, humedad_max, personas_familia, consumo_bidon_pc,
    -- CPC = 94.0 + (2.5*Temp) - (0.8*Humedad) - (5.2*Hab) + (15.0*Bidon) + e
    ROUND(94.0 + (2.5 * temperatura_max) - (0.8 * humedad_max)
               - (5.2 * personas_familia) + (15.0 * consumo_bidon_pc) + ruido, 2) AS cpc_consumo
FROM base;
"""


# -----------------------------------------------------------------------------
# PASO 1: SIMULACIÓN DE DATOS EN DUCKDB
# -----------------------------------------------------------------------------
@st.cache_data
def generar_datos(semilla: float, n: int) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.execute(f"SELECT setseed({semilla});")
    con.execute(SQL_SIMULACION.format(n=int(n)))
    df = con.execute("SELECT * FROM telemetria_agua ORDER BY residencia_id").df()
    con.close()
    return df


@st.cache_resource
def ajustar_modelos(df: pd.DataFrame):
    X = df[VARIABLES_X]
    Y = df[VARIABLE_Y]
    # PASO 3: statsmodels OLS
    modelo_ols = sm.OLS(Y, sm.add_constant(X)).fit()
    # PASO 5: scikit-learn
    model_sk = LinearRegression().fit(X, Y)
    return modelo_ols, model_sk


# -----------------------------------------------------------------------------
# BARRA LATERAL
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Parámetros")
semilla = st.sidebar.number_input(
    "Semilla de DuckDB (setseed)", min_value=0.0, max_value=1.0, value=0.42, step=0.01,
    help="El profesor usa 0.42 para obtener resultados reproducibles.",
)
n_registros = st.sidebar.slider("Número de residencias", 30, 1000, 100, step=10)
alpha = st.sidebar.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Asignatura:** Análisis y Exploración de Datos (INC-2505)  \n"
    "**Unidad 4:** Análisis Estadístico Inferencial  \n"
    "**Subtema:** 4.2 Regresión lineal y correlación  \n"
    "**Caso:** Arellano & Peña (2020), NovaSinergia"
)

# -----------------------------------------------------------------------------
# CÁLCULOS
# -----------------------------------------------------------------------------
df_agua = generar_datos(semilla, n_registros)
modelo_ols, model_sk = ajustar_modelos(df_agua)

X = df_agua[VARIABLES_X]
Y = df_agua[VARIABLE_Y]
matriz_correlacion = df_agua.drop(columns=["residencia_id"]).corr(method="pearson")
residuos = modelo_ols.resid
stat_sw, p_val_sw = stats.shapiro(residuos)
dw_stat = durbin_watson(residuos)
y_pred = model_sk.predict(X)
r2_sk = r2_score(Y, y_pred)
rmse_sk = np.sqrt(mean_squared_error(Y, y_pred))

# -----------------------------------------------------------------------------
# ENCABEZADO
# -----------------------------------------------------------------------------
st.title("💧 Práctica 4.2: Regresión Lineal y Correlación")
st.markdown(
    "Predicción del **Consumo Per Cápita de agua potable (CPC, lt/hab-día)** a partir de la "
    "temperatura máxima, la humedad máxima, el número de personas por familia y el consumo "
    "de agua de garrafón/bidón. Pipeline: **DuckDB → pandas → statsmodels (OLS) → scikit-learn**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", f"{r2_sk * 100:.2f}%")
c2.metric("R² ajustado", f"{modelo_ols.rsquared_adj:.3f}")
c3.metric("RMSE", f"{rmse_sk:.4f}")
c4.metric("Prob (F)", f"{modelo_ols.f_pvalue:.2e}")

tabs = st.tabs([
    "1️⃣ Datos (DuckDB)",
    "2️⃣ Correlación",
    "3️⃣ Modelo OLS",
    "4️⃣ Supuestos",
    "5️⃣ Métricas",
    "📊 Panel de gráficas",
    "🔮 Simulador",
])

# ---------------------------- TAB 1: DATOS ----------------------------------
with tabs[0]:
    st.subheader("Paso 1: Simulación de datos relacionales en DuckDB")
    st.markdown(
        f"Se genera la tabla `telemetria_agua` con **{n_registros} residencias** usando "
        f"`setseed({semilla})`. Primero se crean las variables (CTE `base`) y después el "
        "consumo se calcula con esas mismas columnas más un error aleatorio de −3 a 3."
    )
    with st.expander("Ver consulta SQL"):
        st.code(SQL_SIMULACION.format(n=int(n_registros)), language="sql")

    st.markdown("**Primeros 5 registros obtenidos de DuckDB**")
    st.dataframe(df_agua.head(), width="stretch", hide_index=True)

    st.markdown("**Estadística descriptiva**")
    st.dataframe(df_agua.drop(columns=["residencia_id"]).describe().round(3), width="stretch")

    with st.expander("Ver tabla completa"):
        st.dataframe(df_agua, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Descargar datos (CSV)", df_agua.to_csv(index=False).encode("utf-8"),
        file_name="telemetria_agua.csv", mime="text/csv",
    )

# ---------------------------- TAB 2: CORRELACIÓN -----------------------------
with tabs[1]:
    st.subheader("Paso 2: Matriz de correlación de Pearson (r)")
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, cbar=True, ax=ax)
        ax.set_title("Matriz de Correlación de Pearson (r)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        st.markdown("**Matriz numérica**")
        st.dataframe(matriz_correlacion.round(4), width="stretch")

        corr_y = matriz_correlacion[VARIABLE_Y].drop(VARIABLE_Y)
        st.markdown("**Correlación de cada variable con `cpc_consumo`**")

        def fuerza(r):
            a = abs(r)
            if a >= 0.7: return "fuerte"
            if a >= 0.4: return "moderada"
            if a >= 0.2: return "débil"
            return "muy débil / nula"

        tabla_corr = pd.DataFrame({
            "r": corr_y.round(4),
            "Dirección": np.where(corr_y > 0, "positiva", "negativa"),
            "Intensidad": [fuerza(r) for r in corr_y],
        })
        st.dataframe(tabla_corr, width="stretch")

    st.subheader("Diagramas de dispersión contra el consumo")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, var in zip(axes, VARIABLES_X):
        sns.regplot(data=df_agua, x=var, y=VARIABLE_Y, ax=ax,
                    scatter_kws={"alpha": 0.6}, line_kws={"color": "red", "linewidth": 2})
        ax.set_title(f"{var}  (r = {corr_y[var]:.2f})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "La humedad tiene la correlación individual más fuerte (negativa): a mayor humedad, "
        "menor consumo. Ninguna variable explica sola el consumo; por eso se ajusta un modelo "
        "de regresión **múltiple**. Las correlaciones entre predictores son cercanas a 0, "
        "así que no hay multicolinealidad."
    )

# ---------------------------- TAB 3: OLS -------------------------------------
with tabs[2]:
    st.subheader("Paso 3: Regresión lineal múltiple con statsmodels (OLS)")
    coefs = pd.DataFrame({
        "coef": modelo_ols.params,
        "std err": modelo_ols.bse,
        "t": modelo_ols.tvalues,
        "P>|t|": modelo_ols.pvalues,
        "[0.025": modelo_ols.conf_int()[0],
        "0.975]": modelo_ols.conf_int()[1],
    })
    coefs["¿Significativo?"] = np.where(coefs["P>|t|"] < alpha, f"Sí (p < {alpha})", "No")
    st.dataframe(coefs.round(4), width="stretch")

    reales = {"const": 94.0, "temperatura_max": 2.5, "humedad_max": -0.8,
              "personas_familia": -5.2, "consumo_bidon_pc": 15.0}
    comp = pd.DataFrame({
        "Valor real (simulación)": pd.Series(reales),
        "Estimado (OLS)": modelo_ols.params.round(4),
    })
    comp["Diferencia"] = (comp["Estimado (OLS)"] - comp["Valor real (simulación)"]).round(4)
    st.markdown("**Comparación de coeficientes estimados vs. los usados para simular**")
    st.dataframe(comp, width="stretch")

    with st.expander("Ver resumen completo de statsmodels (`modelo_ols.summary()`)"):
        st.text(modelo_ols.summary().as_text())

    st.markdown("**Interpretación de los coeficientes**")
    p = modelo_ols.params
    st.markdown(
        f"- **Temperatura:** por cada °C adicional, el consumo sube **{p['temperatura_max']:.2f}** lt/hab-día.\n"
        f"- **Humedad:** por cada punto % adicional, el consumo baja **{abs(p['humedad_max']):.2f}** lt/hab-día.\n"
        f"- **Personas por familia:** cada integrante extra reduce el consumo per cápita en **{abs(p['personas_familia']):.2f}** lt/hab-día (economía de escala en el hogar).\n"
        f"- **Consumo de bidón:** cada litro de garrafón por persona se asocia con **{p['consumo_bidon_pc']:.2f}** lt/hab-día más.\n"
        f"- La prueba F global tiene p = {modelo_ols.f_pvalue:.2e}, así que el modelo en conjunto es significativo."
    )

# ---------------------------- TAB 4: SUPUESTOS -------------------------------
with tabs[3]:
    st.subheader("Paso 4: Diagnóstico de residuos y supuestos")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**1. Normalidad (Shapiro-Wilk)**")
        st.write(f"Estadístico = **{stat_sw:.4f}**, valor p = **{p_val_sw:.4f}**")
        if p_val_sw > alpha:
            st.success(f"Los residuos cumplen el supuesto de NORMALIDAD (p > {alpha}).")
        else:
            st.error(f"Se rechaza la normalidad de los residuos (p ≤ {alpha}).")
    with s2:
        st.markdown("**2. Independencia (Durbin-Watson)**")
        st.write(f"Estadístico = **{dw_stat:.4f}**")
        if 1.5 <= dw_stat <= 2.5:
            st.success("Valor entre 1.5 y 2.5: no hay autocorrelación en los residuos.")
        else:
            st.warning("Valor fuera de 1.5–2.5: posible autocorrelación.")

    g1, g2, g3 = st.columns(3)
    with g1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(y_pred, residuos, color="purple", alpha=0.6)
        ax.axhline(0, color="r", linestyle="--", lw=2)
        ax.set_title("Residuos vs Ajustados (Homocedasticidad)")
        ax.set_xlabel("Valores predichos (ŷ)")
        ax.set_ylabel("Residuos (e = y - ŷ)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sm.qqplot(residuos, line="45", fit=True, ax=ax)
        ax.set_title("Gráfico Q-Q Normal (Residuos)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with g3:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.histplot(residuos, kde=True, color="teal", ax=ax)
        ax.set_title("Histograma de residuos")
        ax.set_xlabel("Residuo")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.caption(
        "Los residuos se reparten al azar alrededor de 0 sin forma de embudo (homocedasticidad), "
        "siguen la línea del Q-Q (normalidad) y el Durbin-Watson ≈ 2 (independencia). "
        "El error se simuló como uniforme de −3 a 3, por eso el histograma es algo plano."
    )

# ---------------------------- TAB 5: MÉTRICAS --------------------------------
with tabs[4]:
    st.subheader("Paso 5: Métricas de rendimiento (scikit-learn)")
    m1, m2 = st.columns(2)
    m1.metric("Coeficiente de determinación (R²)", f"{r2_sk * 100:.2f}%")
    m2.metric("Error cuadrático medio raíz (RMSE)", f"{rmse_sk:.4f} lt/hab-día")

    b = model_sk.coef_
    st.markdown("**Ecuación del modelo ajustado**")
    st.latex(
        rf"\widehat{{CPC}} = {model_sk.intercept_:.2f} + {b[0]:.2f}\,T_{{max}} "
        rf"{b[1]:+.2f}\,H_{{max}} {b[2]:+.2f}\,Personas {b[3]:+.2f}\,Bid\acute{{o}}n"
    )
    st.markdown(
        f"El modelo explica el **{r2_sk * 100:.1f}%** de la variación del consumo per cápita. "
        f"En promedio, la predicción se equivoca por unos **{rmse_sk:.2f} lt/hab-día**."
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, Y, alpha=0.7, color="teal")
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], "r--", lw=2)
    ax.set_title(f"Valores reales vs predichos (R² = {r2_sk * 100:.1f}%)")
    ax.set_xlabel("CPC predicho (ŷ)")
    ax.set_ylabel("CPC real (y)")
    plt.tight_layout()
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.pyplot(fig)
    plt.close(fig)

# ---------------------------- TAB 6: PANEL -----------------------------------
with tabs[5]:
    st.subheader("Paso 6: Panel de diagnóstico visual (código del profesor)")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)",
                 fontsize=16, fontweight="bold")

    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
    axes[0, 0].set_title("1. Matriz de Correlación de Pearson (r)")

    sns.regplot(data=df_agua, x="humedad_max", y="cpc_consumo", color="b", ax=axes[0, 1],
                line_kws={"color": "red", "linewidth": 2})
    axes[0, 1].set_title("2. Tendencia: Humedad Máx vs Consumo")
    axes[0, 1].set_xlabel("Humedad Máxima (%)")
    axes[0, 1].set_ylabel("CPC Consumo (lt/hab-día)")

    axes[0, 2].scatter(y_pred, Y, alpha=0.7, color="teal")
    axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], "r--", lw=2)
    axes[0, 2].set_title(f"3. Val. Reales vs Predichos (R² = {r2_sk * 100:.1f}%)")
    axes[0, 2].set_xlabel("CPC Predicho (ŷ)")
    axes[0, 2].set_ylabel("CPC Real (y)")

    axes[1, 0].scatter(y_pred, residuos, color="purple", alpha=0.6)
    axes[1, 0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_title("4. Residuos vs Ajustados (Homocedasticidad)")
    axes[1, 0].set_xlabel("Valores Predichos (ŷ)")
    axes[1, 0].set_ylabel("Residuos (e = y - ŷ)")

    sm.qqplot(residuos, line="45", fit=True, ax=axes[1, 1])
    axes[1, 1].set_title("5. Gráfico Q-Q Normal (Residuos)")

    sns.histplot(residuos, kde=True, color="teal", ax=axes[1, 2])
    axes[1, 2].set_title("6. Distribución de Residuos")
    axes[1, 2].set_xlabel("Residuo")

    plt.tight_layout()
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    st.download_button("⬇️ Descargar panel (PNG)", buf.getvalue(),
                       file_name="graficas_practica4_2.png", mime="image/png")

# ---------------------------- TAB 7: SIMULADOR -------------------------------
with tabs[6]:
    st.subheader("🔮 Simulador: predice el consumo de una vivienda")
    st.markdown("Mueve los valores y el modelo ajustado calcula el consumo per cápita esperado.")
    p1, p2 = st.columns(2)
    with p1:
        t_in = st.slider("Temperatura máxima (°C)", 18.0, 32.0, 25.0, 0.5)
        h_in = st.slider("Humedad máxima (%)", 50.0, 90.0, 70.0, 1.0)
    with p2:
        f_in = st.slider("Personas en la familia", 3, 7, 4)
        b_in = st.slider("Consumo de bidón (lt/hab-día)", 0.2, 2.0, 1.0, 0.05)

    entrada = pd.DataFrame([[t_in, h_in, f_in, b_in]], columns=VARIABLES_X)
    pred = model_sk.predict(entrada)[0]
    pred_ols = modelo_ols.get_prediction(sm.add_constant(entrada, has_constant="add")).summary_frame(alpha=alpha)
    lo, hi = pred_ols["obs_ci_lower"].iloc[0], pred_ols["obs_ci_upper"].iloc[0]

    st.metric("Consumo per cápita estimado", f"{pred:.2f} lt/hab-día")
    st.write(f"Intervalo de predicción al {int((1 - alpha) * 100)}%: **{lo:.2f} – {hi:.2f}** lt/hab-día")
    st.write(f"Consumo total estimado de la vivienda: **{pred * f_in:.1f} lt/día**")

st.markdown("---")
st.caption("Práctica 4.2 · Análisis y Exploración de Datos · Stack: DuckDB, pandas, statsmodels, scikit-learn, matplotlib, seaborn, Streamlit")
