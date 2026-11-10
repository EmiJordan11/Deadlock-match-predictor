"""Genera la aplicación de la Entrega 4 en docs/ a partir de los datos del pipeline.

Uso, desde la raíz del repositorio:

    python app/build_app.py

Qué hace, en orden:

1. Lee los cortes que deja el DAG `deadlock_ingest` en include/output/silver/ y el bronce
   de partidas, para contar qué se filtró en cada paso.
2. Entrena el modelo elegido en la Entrega 3 (notebooks/entrega3_modelado_v3.ipynb, sección 5):
   una regresión logística sobre las 38 columnas de composición, sin escalar, multa l2 y
   C = 0,1, con el último corte de entrenamiento. Mide de nuevo, con el mismo walk-forward,
   la tabla de comparación que muestra la aplicación.
3. Verifica que las predicciones coincidan con el modelo que guardó el notebook
   (notebooks/modelo_entrega3_composicion.joblib).
4. Arma los gráficos con Altair y escribe docs/index.html, que GitHub Pages publica tal cual.
   Baja una sola vez los retratos de los héroes y las librerías de Vega a docs/.

Si los datos cambian (otro corte, más partidas), correrlo de nuevo actualiza la aplicación.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import pickle
import sys
import urllib.request
import warnings
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent
PLANTILLA = RAIZ / "app" / "plantilla.html"
SALIDA = RAIZ / "docs"
PLATA = RAIZ / "include" / "output" / "silver"
BRONCE = RAIZ / "include" / "output" / "bronze"
HEROES_API = RAIZ / "include" / "output" / "heroes_api_2026-10-26.json"
MODELO_NOTEBOOK = RAIZ / "notebooks" / "modelo_entrega3_composicion.joblib"
NOTEBOOK_E2 = RAIZ / "notebooks" / "entrega2_eda_hipotesis.ipynb"
sys.path.insert(0, str(RAIZ / "include"))
from deadlock import schema  # noqa: E402
from deadlock.transform import parse_match  # noqa: E402

SEMILLA = 42
np.random.seed(SEMILLA)

# ---- lo que se eligió en la Entrega 3 (notebook v3) -------------------------------------
C_ELEGIDO = 0.1          # sección 5.4: sin escalar, multa l2
VENTANA = 1              # sección 5.5: se entrena con el último corte de entrenamiento
C_E3_ANTERIOR = 0.01     # el modelo presentado en la Entrega 3 (winrates, 3 columnas), como referencia
C_CRUCES = 0.001         # sección 6.2: la mejor C para la composición con los cruces

# ---- las hipótesis de la Entrega 2 (notebooks/entrega2_eda_hipotesis.ipynb, sección 7) ----
# Se midieron sobre el corte de esa entrega; build_app comprueba que sigan escritas en el notebook.
ENTREGA_2 = {"h1": "0,374", "h3_counter": "0,998", "h3_synergy": "0,981", "h4": "0,025",
             "e2_corte": "16/09, 30.000 partidas"}

# ---- versiones de las librerías que dibujan los gráficos (las que usa Altair 6) ---------------
VENDOR = {"vega.min.js": "https://cdn.jsdelivr.net/npm/vega@6.4.0/build/vega.min.js",
          "vega-lite.min.js": "https://cdn.jsdelivr.net/npm/vega-lite@6.4.1/build/vega-lite.min.js",
          "vega-embed.min.js": "https://cdn.jsdelivr.net/npm/vega-embed@7.2.0/build/vega-embed.min.js"}

# ---- la paleta, la misma que el CSS de la plantilla -------------------------------------
TEXT, MUTED, LINE, GRID = "#ece2cc", "#a3977f", "#4a3f30", "#2a231b"
BRASS, AMBER, SAPPHIRE, MEDIO, FONDO = "#c9a46a", "#e8922e", "#4f93e3", "#8f8473", "#17130f"
FUENTE = "Barlow Condensed"
COMP = schema.COMPOSICION
WINRATES = ["hero_winrate_adv_mean", "counter_winrate_adv_mean", "synergy_winrate_adv_mean"]
REGLA = ["hero_winrate_adv_mean"]
ORDEN_DRAFT = "ABBAABBAABBA"
NUMEROS = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve"}
avisos: list[str] = []


# ---- formato en castellano ---------------------------------------------------------------------
def num(x: float, d: int = 1) -> str:
    s = f"{abs(x):,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return ("−" if x < 0 and round(abs(x), d) > 0 else "") + s


def pct(x: float, d: int = 1) -> str:
    """x entre 0 y 1."""
    return num(100 * x, d) + " %"


def entero(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fecha(corte: str, anio: bool = False) -> str:
    return f"{corte[8:10]}/{corte[5:7]}" + (f"/{corte[:4]}" if anio else "")


def frase_fraccion(p: float) -> str:
    """Un porcentaje como lo diría una persona, si está cerca de una fracción simple."""
    for v, t in ((100 / 3, "una de cada tres"), (50, "la mitad de las"), (200 / 3, "dos de cada tres"), (75, "tres de cada cuatro")):
        if abs(p - v) < 2:
            return t
    return f"{round(p)} de cada 100"


def frase_corta(p: float) -> str:
    for v, t in ((200 / 3, "2 de cada 3"), (75, "3 de cada 4"), (60, "3 de cada 5")):
        if abs(p - v) < 1.5:
            return t
    return f"{round(p)} de cada 100"


def descargar(url: str, destino: Path) -> None:
    if destino.exists():
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (build_app, proyecto academico)"})
    destino.write_bytes(urllib.request.urlopen(req, timeout=60).read())


# ---- pasos del pipeline (los mismos que el notebook) -----------------------------------------
class Columnas(BaseEstimator, TransformerMixin):
    """Se queda con algunas columnas: el primer paso de cada pipeline, como ElegirColumnas en el notebook."""

    def __init__(self, columnas=()):
        self.columnas = columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columnas)]


class Cruces(BaseEstimator, TransformerMixin):
    """La composición más una columna por cada cruce de héroes: +1 si el primero juega en A y el segundo en B."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        comp = np.asarray(X, dtype=float)
        en_a, en_b = (comp == 1).astype(float), (comp == -1).astype(float)
        i, j = np.triu_indices(comp.shape[1], 1)
        return np.hstack([comp, en_a[:, i] * en_b[:, j] - en_b[:, i] * en_a[:, j]])


class VentanaDeslizante:
    """Walk-forward: entrena con los m cortes anteriores y valida con el siguiente."""

    def __init__(self, m=1):
        self.m = m

    def split(self, X, y=None, groups=None):
        g = np.asarray(groups)
        orden = np.sort(np.unique(g))
        for i in range(self.m, len(orden)):
            yield np.flatnonzero(np.isin(g, orden[i - self.m:i])), np.flatnonzero(g == orden[i])

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(np.unique(groups)) - self.m


def armar(modelo, columnas, escalar=True):
    return Pipeline([("elegir", Columnas(tuple(columnas))), ("escalar", StandardScaler() if escalar else "passthrough"), ("modelo", modelo)])


def elegido():
    return armar(LogisticRegression(max_iter=5000, C=C_ELEGIDO), COMP, escalar=False)


# =====================================================================================
# 1. Los datos
# =====================================================================================
print("· leyendo los cortes de", PLATA.relative_to(RAIZ))
partes = []
for carpeta in sorted(PLATA.glob("corte=*")):
    t = pd.read_parquet(carpeta / "features.parquet")
    t["corte"] = carpeta.name.split("=")[1]
    partes.append(t)
df = pd.concat(partes, ignore_index=True)
assert df["match_id"].is_unique, "una partida aparece en dos cortes"
y = df["winner"].eq("A").astype(int)
CORTES = sorted(df["corte"].unique())
CORTE_TEST = CORTES[-1]
es_test = df["corte"].eq(CORTE_TEST)
anteriores = int(sum((pd.to_datetime(t["match_date"]) < pd.Timestamp(c)).sum() for c, t in df.groupby("corte")))
assert anteriores == 0, "hay partidas anteriores a su corte: las ventanas se pisan"

X_train, y_train, g_train = df[~es_test], y[~es_test], df.loc[~es_test, "corte"]
X_test, y_test = df[es_test], y[es_test].to_numpy()
ultimos = sorted(g_train.unique())[-VENTANA:]
X_final, y_final = X_train[g_train.isin(ultimos)], y_train[g_train.isin(ultimos)]
FECHA_TRAIN = fecha(ultimos[-1])

# el embudo: el bronce de partidas pasado de nuevo por el filtro de calidad del DAG
def motivo(raw: dict) -> str:
    eq = {"Team0": [], "Team1": []}
    for j in raw.get("players") or []:
        if j.get("team") in eq and j.get("hero_id") is not None:
            eq[j["team"]].append(j["hero_id"])
    if len(eq["Team0"]) != 6 or len(eq["Team1"]) != 6:
        return "incompleto"
    if any(h not in schema.HERO_IDS for h in eq["Team0"] + eq["Team1"]):
        return "catalogo"
    if len(set(eq["Team0"] + eq["Team1"])) != 12:
        return "repetido"
    if raw.get("winning_team") not in ("Team0", "Team1"):
        return "ganador"
    return "ok"


print("· contando el filtro de calidad sobre el bronce")
embudo = Counter()
for c in CORTES:
    for pag in sorted((BRONCE / f"corte={c}" / "matches").glob("*.json.gz")):
        with gzip.open(pag, "rt", encoding="utf-8") as f:
            for raw in json.load(f):
                m = motivo(raw)
                assert (m == "ok") == (parse_match(raw) is not None), "el conteo no coincide con parse_match"
                embudo["leidas"] += 1
                embudo[m] += 1
por_corte = df.groupby("corte").size()

# =====================================================================================
# 2. El modelo
# =====================================================================================
print("· entrenando el modelo elegido con", ", ".join(ultimos))
modelo = elegido().fit(X_final, y_final)
lr = modelo.named_steps["modelo"]
w = pd.Series(lr.coef_[0], index=COMP)
b0 = float(lr.intercept_[0])
sig = lambda z: 1 / (1 + np.exp(-z))
p_test = modelo.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, p_test)
acc_test = accuracy_score(y_test, p_test >= 0.5)
piso = DummyClassifier(strategy="most_frequent").fit(X_final[REGLA], y_final)
piso_clase = "A" if piso.predict(X_test[REGLA].iloc[:1])[0] == 1 else "B"
piso_acc = accuracy_score(y_test, piso.predict(X_test[REGLA]))

coincide = None
if MODELO_NOTEBOOK.exists():
    import joblib
    paquete = joblib.load(MODELO_NOTEBOOK)
    p_nb = paquete["modelo"].predict_proba(X_test[paquete["columnas"]])[:, 1]
    coincide = float(np.abs(p_nb - p_test).max())
    if coincide > 1e-9:
        avisos.append(f"las predicciones no coinciden con el modelo del notebook (diferencia máxima {coincide:.2e}): "
                      "¿cambiaron los datos? Volvé a correr el notebook v3")
else:
    avisos.append(f"no está {MODELO_NOTEBOOK.relative_to(RAIZ)}: no se pudo comparar con el notebook")

# la tabla de comparación, con el mismo walk-forward de la Entrega 3
cv = VentanaDeslizante(VENTANA)


def medir(pipe, cv=cv):
    r = cross_validate(pipe, X_train, y_train, groups=g_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return r["test_score"].mean(), r["test_score"].std()


print("· midiendo la tabla de comparación (walk-forward)")
ELEGIDO_NOMBRE = "Logística sobre la composición · elegida"
candidatos = {
    "Piso": (armar(DummyClassifier(strategy="most_frequent"), REGLA), "—"),
    "Regla de una columna (winrate de la API)": (armar(LogisticRegression(), REGLA), "1"),
    "Logística con los winrates de la API": (armar(LogisticRegression(max_iter=5000, C=10), WINRATES), "3"),
    ELEGIDO_NOMBRE: (elegido(), str(len(COMP))),
    "Random forest sobre la composición": (armar(RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=1, random_state=SEMILLA), COMP), str(len(COMP))),
    "Gradient boosting sobre la composición": (armar(HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_leaf_nodes=7, random_state=SEMILLA), COMP), str(len(COMP))),
    "Árbol de decisión sobre la composición": (armar(DecisionTreeClassifier(max_depth=10, min_samples_leaf=100, random_state=SEMILLA), COMP), str(len(COMP))),
}
comparacion = {n: (*medir(p), cols) for n, (p, cols) in candidatos.items()}
auc_cv, sd_cv, _ = comparacion[ELEGIDO_NOMBRE]
mejor_otro = max((v[0] for n, v in comparacion.items() if n != ELEGIDO_NOMBRE), default=0)
if mejor_otro - auc_cv > sd_cv:
    avisos.append("otro modelo le gana a la composición por más que el desvío: revisar la sección del modelo")

azar = medir(elegido(), cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA))[0] - auc_cv
por_C = [medir(armar(LogisticRegression(max_iter=5000, C=c), COMP, escalar=False))[0] for c in (0.01, 0.1, 1, 10)]
auc_cruces = medir(Pipeline([("elegir", Columnas(tuple(COMP))), ("cruces", Cruces()), ("modelo", LogisticRegression(max_iter=5000, C=C_CRUCES))]))[0]
modelo_e3 = armar(LogisticRegression(max_iter=5000, C=C_E3_ANTERIOR), WINRATES).fit(X_final, y_final)
auc_e3 = roc_auc_score(y_test, modelo_e3.predict_proba(X_test)[:, 1])

# =====================================================================================
# 3. Los héroes
# =====================================================================================
api = {h["id"]: h for h in json.load(open(HEROES_API, encoding="utf-8"))}
with open(PLATA / f"corte={ultimos[-1]}" / "stat_tables.pkl", "rb") as f:
    adv_api = pickle.load(f)["hero"]
heroes = []
for hid, slug in schema.HEROES:
    col = f"hero_{slug}"
    en = df[col].ne(0)
    gano = (df[col].eq(1) & y.eq(1)) | (df[col].eq(-1) & y.eq(0))
    imgs = api[hid]["images"]
    descargar(imgs["icon_hero_card_webp"], SALIDA / "heroes" / f"{slug}_card.webp")
    descargar(imgs["icon_image_small_webp"], SALIDA / "heroes" / f"{slug}_sm.webp")
    heroes.append({"slug": slug, "nombre": api[hid]["name"], "w": round(float(w[col]), 6),
                   "pts": round(float((sig(w[col]) - 0.5) * 100), 2), "api": round(50 + 100 * adv_api.get(hid, 0.0), 2),
                   "partidas": int(en.sum()), "wr": round(float(100 * gano[en].mean()), 2),
                   "card": f"heroes/{slug}_card.webp", "sm": f"heroes/{slug}_sm.webp"})
H = pd.DataFrame(heroes).set_index("slug")
r_api = float(np.corrcoef(H["w"], H["api"])[0, 1])
mas, menos = H["pts"].nlargest(2), H["pts"].nsmallest(2)

# =====================================================================================
# 4. Los gráficos (Altair)
# =====================================================================================
ES = lambda lo, hi, **k: alt.Scale(domain=[lo, hi], zero=False, nice=False, **k)
mitad = lambda lo, hi: alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color=BRASS, strokeDash=[5, 5], opacity=0.7).encode(y=alt.Y("y:Q", scale=ES(lo, hi)))

# tallo 1 · los datos: cuánto gana A en cada fecha
g = pd.DataFrame({"corte": df["corte"], "ganaA": y}).groupby("corte")["ganaA"].agg(["mean", "size"]).reset_index()
g["fecha"] = g["corte"].map(fecha)
g["pct"] = 100 * g["mean"]
g["ic"] = 196 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
g["lo"], g["hi"] = g["pct"] - g["ic"], g["pct"] + g["ic"]
g["uso"] = np.where(g["corte"].eq(CORTE_TEST), "test", "entrena")
g["uso_largo"] = np.where(g["corte"].eq(CORTE_TEST), "test (se mira una vez)", "entrenamiento y validación")
datos_cortes = g[["fecha", "pct", "lo", "hi", "size", "uso", "uso_largo"]].round(2)
if not datos_cortes["pct"].between(47, 53).all():
    avisos.append("en alguna fecha A gana fuera de 47–53 %: revisar el título de «Los datos»")
eje_y = alt.Y("pct:Q", title="% de partidas que ganó el Equipo A", scale=ES(46, 54))
base = alt.Chart(datos_cortes).encode(x=alt.X("fecha:O", title=None, sort=[fecha(c) for c in CORTES], axis=alt.Axis(labelAngle=0, labelFontSize=15)))
spec_datos = (
    mitad(46, 54)
    + base.mark_rule(color=AMBER, strokeWidth=2, opacity=0.6).encode(y=alt.Y("lo:Q", scale=ES(46, 54)), y2="hi:Q")
    + base.mark_point(shape="diamond", filled=True, size=220, color=AMBER).encode(
        y=eje_y, tooltip=[alt.Tooltip("fecha:N", title="Fecha"), alt.Tooltip("size:Q", title="Partidas", format=",.0f"),
                          alt.Tooltip("pct:Q", title="Ganó A (%)", format=".1f"), alt.Tooltip("uso_largo:N", title="Se usa para")])
    + base.mark_text(dy=-18, color=TEXT, font=FUENTE, fontSize=14).encode(y=eje_y, text=alt.Text("pct:Q", format=".1f"))
    + base.mark_text(color=MUTED, font=FUENTE, fontSize=12).encode(y=alt.datum(46.6), text="uso:N")
).properties(width="container", height=300)

# tallo 2 · lo que encontramos: cuánto gana A según la ventaja de sus héroes
d = pd.DataFrame({"ganaA": y, "decil": pd.qcut(df["hero_winrate_adv_mean"], 10, labels=list(range(1, 11)))})
dec = d.groupby("decil", observed=True)["ganaA"].agg(["mean", "size"]).reset_index()
dec["pct"] = 100 * dec["mean"]
dec["ic"] = 196 * np.sqrt(dec["mean"] * (1 - dec["mean"]) / dec["size"])
dec["lo"], dec["hi"] = dec["pct"] - dec["ic"], dec["pct"] + dec["ic"]
dec["decil"] = dec["decil"].astype(int)
dec = dec[["decil", "pct", "lo", "hi", "size"]].round(2)
color_ventaja = alt.Color("pct:Q", scale=alt.Scale(domain=[34, 50, 66], range=[SAPPHIRE, MEDIO, AMBER]), legend=None)
x_dec = alt.X("decil:O", title=["Ventaja de los héroes de A según la API, en deciles", "1 = A mucho más débil · 10 = A mucho más fuerte"],
              axis=alt.Axis(labelAngle=0, labelFontSize=14, titlePadding=12))
y_dec = alt.Y("pct:Q", title="% de partidas que ganó el Equipo A", scale=ES(28, 72))
bd = alt.Chart(dec).encode(x=x_dec)
spec_hallazgo = (
    mitad(28, 72)
    + bd.mark_line(color=LINE, strokeWidth=2).encode(y=y_dec)
    + bd.mark_rule(strokeWidth=2, opacity=0.6).encode(y=alt.Y("lo:Q", scale=ES(28, 72)), y2="hi:Q", color=color_ventaja)
    + bd.mark_point(shape="diamond", filled=True, size=200, opacity=1).encode(
        y=y_dec, color=color_ventaja,
        tooltip=[alt.Tooltip("decil:O", title="Decil"), alt.Tooltip("size:Q", title="Partidas", format=",.0f"),
                 alt.Tooltip("pct:Q", title="Ganó A (%)", format=".1f")])
    + bd.mark_text(dy=-18, color=TEXT, font=FUENTE, fontSize=13).encode(y=y_dec, text=alt.Text("pct:Q", format=".0f"))
).properties(width="container", height=320)

# tallo 3 · el modelo: cuánto acierta según qué tan despareja es la composición
conf = np.abs(p_test - 0.5)
# el «|» parte cada rótulo en dos renglones, para que entren en una pantalla angosta
q = pd.qcut(conf, 5, labels=["20 % más|parejas", "2.º|quinto", "3.º|quinto", "4.º|quinto", "20 % más|desparejas"])
quint = pd.DataFrame({"tramo": q, "ok": (p_test >= 0.5) == y_test}).groupby("tramo", observed=True)["ok"].agg(["mean", "size"]).reset_index()
quint["pct"], quint["piso"], quint["orden"] = 100 * quint["mean"], 100 * piso_acc, range(len(quint))
quint["tramo"] = quint["tramo"].astype(str)
quint["nombre"] = quint["tramo"].str.replace("|", " ", regex=False)
quint = quint[["tramo", "nombre", "pct", "piso", "size", "orden"]].round(2)
x_q = alt.X("tramo:N", sort=alt.SortField("orden"), title="Partidas del test, de la composición más pareja a la más despareja",
            axis=alt.Axis(labelAngle=0, labelFontSize=14, titlePadding=12, labelExpr="split(datum.label, '|')"))
y_q = alt.Y("pct:Q", title="% de partidas en que acierta quién gana", scale=ES(45, 72))
bq = alt.Chart(quint).encode(x=x_q)
refs = pd.DataFrame({"y": [100 * piso_acc, 100 * acc_test], "ty": [69.2, 71.2],
                     "t": [f"– – piso: {pct(piso_acc)}", f"– – el modelo, en total: {pct(acc_test)}"], "c": [MUTED, BRASS]})
spec_modelo = (
    alt.Chart(refs).mark_rule(strokeDash=[5, 5], opacity=0.8).encode(y=alt.Y("y:Q", scale=ES(45, 72)), color=alt.Color("c:N", scale=None))
    + alt.Chart(refs).mark_text(align="left", x=8, font=FUENTE, fontSize=13).encode(y=alt.Y("ty:Q", scale=ES(45, 72)), text="t:N", color=alt.Color("c:N", scale=None))
    + bq.mark_rule(color=BRASS, strokeWidth=3).encode(y=alt.Y("piso:Q", scale=ES(45, 72)), y2="pct:Q")
    + bq.mark_point(shape="diamond", filled=True, size=260, color=BRASS, opacity=1).encode(
        y=y_q, tooltip=[alt.Tooltip("nombre:N", title="Partidas"), alt.Tooltip("size:Q", title="Cuántas", format=",.0f"),
                        alt.Tooltip("pct:Q", title="Acierta (%)", format=".1f")])
    + bq.mark_text(dy=-20, color=TEXT, font=FUENTE, fontSize=15, fontWeight=600).encode(y=y_q, text=alt.Text("pct:Q", format=".1f"))
).properties(width="container", height=320)

# copa · explorá: un héroe por punto, filtrable por fecha y nivel de juego
TRAMOS = ["hasta 40", "41 a 60", "61 a 80", "más de 80"]
tramo = pd.cut(df["average_badge"], [-1, 40, 60, 80, 1000], labels=TRAMOS).astype(str)
filas = []
for h in heroes:
    col = f"hero_{h['slug']}"
    en = df[col].ne(0)
    gano = (df[col].eq(1) & y.eq(1)) | (df[col].eq(-1) & y.eq(0))
    t = pd.DataFrame({"fecha": df.loc[en, "corte"].map(fecha), "nivel": tramo[en], "gano": gano[en].astype(int)})
    for (fe, ni), a in t.groupby(["fecha", "nivel"])["gano"].agg(["sum", "size"]).iterrows():
        filas.append({"slug": h["slug"], "nombre": h["nombre"], "img": h["sm"], "pts": h["pts"], "api": h["api"],
                      "fecha": fe, "nivel": ni, "g": int(a["sum"]), "n": int(a["size"])})
largo = pd.DataFrame(filas)
fecha_p = alt.param(name="fecha", value="Todas", bind=alt.binding_select(options=["Todas"] + [fecha(c) for c in CORTES], name="Fecha"))
nivel_p = alt.param(name="nivel", value="Todos", bind=alt.binding_select(options=["Todos"] + TRAMOS, name="Nivel de juego"))
X_E = alt.X("pts:Q", title="Puntos que le suma a su equipo según el modelo", scale=ES(-7.5, 8.5))
Y_E = alt.Y("wr:Q", title="% de partidas que ganó su equipo", scale=ES(38, 62, clamp=True))
agregado = (alt.Chart(largo)
            .transform_filter("fecha == 'Todas' || datum.fecha == fecha")
            .transform_filter("nivel == 'Todos' || datum.nivel == nivel")
            .transform_aggregate(g="sum(g)", n="sum(n)", groupby=["slug", "nombre", "img", "pts", "api"])
            .transform_calculate(wr="100 * datum.g / datum.n"))
tt = [alt.Tooltip("nombre:N", title="Héroe"), alt.Tooltip("pts:Q", title="Según el modelo (pts)", format="+.1f"),
      alt.Tooltip("wr:Q", title="Ganó su equipo (%)", format=".1f"), alt.Tooltip("n:Q", title="Partidas", format=",.0f"),
      alt.Tooltip("api:Q", title="Winrate API, 14 días (%)", format=".1f")]
spec_explora = (
    alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color=LINE, strokeDash=[4, 4]).encode(y=alt.Y("y:Q", scale=ES(38, 62)))
    + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=LINE, strokeDash=[4, 4]).encode(x=alt.X("x:Q", scale=ES(-7.5, 8.5)))
    + agregado.mark_circle(size=1500, opacity=1, stroke=BRASS, strokeWidth=1.5, fill=FONDO).encode(x=X_E, y=Y_E, tooltip=tt)
    + agregado.mark_image(width=34, height=34, cursor="pointer").encode(x=X_E, y=Y_E, url="img:N", tooltip=tt)
).add_params(fecha_p, nivel_p).properties(width="container", height=460)

# copa · el draft: la probabilidad pick a pick (los datos los pone la página)
curva = alt.Chart(alt.NamedData(name="curva"))
X_C = alt.X("pick:Q", title="Pick", scale=alt.Scale(domain=[0, 12]), axis=alt.Axis(values=list(range(13)), format="d"))
Y_C = alt.Y("pa:Q", title="Probabilidad de que gane A (%)", scale=ES(20, 80))
spec_curva = (
    mitad(20, 80)
    + curva.mark_line(color=MUTED, strokeWidth=2).encode(x=X_C, y=Y_C)
    + curva.mark_point(shape="diamond", filled=True, size=150, opacity=1).encode(
        x=X_C, y=Y_C, color=alt.Color("equipo:N", scale=alt.Scale(domain=["A", "B", "—"], range=[AMBER, SAPPHIRE, MUTED]), legend=None),
        tooltip=[alt.Tooltip("pick:Q", title="Pick"), alt.Tooltip("heroe:N", title="Héroe"),
                 alt.Tooltip("equipo:N", title="Eligió"), alt.Tooltip("pa:Q", title="Gana A (%)", format=".1f")])
).properties(width="container", height=230)


def config(ch):
    return (ch.configure(background="transparent", font=FUENTE, padding={"left": 4, "right": 12, "top": 10, "bottom": 4})
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=LINE, tickColor=LINE,
                            labelFontSize=13, titleFontSize=13, titleFontWeight="normal", labelFont=FUENTE, titleFont=FUENTE))


SPECS = {k: config(v).to_dict() for k, v in {"g-datos": spec_datos, "g-hallazgo": spec_hallazgo, "g-modelo": spec_modelo,
                                               "g-explora": spec_explora, "g-curva": spec_curva}.items()}

# =====================================================================================
# 5. La página
# =====================================================================================
test = X_test.reset_index(drop=True).assign(p=p_test, real=y_test)
test["ok"] = (test["p"] >= 0.5) == test["real"].astype(bool)
idx = [0, int(test[test["ok"]]["p"].idxmax()), int(test[~test["ok"]]["p"].idxmax()),
       int((test["p"] - 0.5).abs().idxmin()), int(test[test["ok"]]["p"].idxmin())]
ejemplos = [{"a": [k[5:] for k in COMP if test.loc[i, k] == 1], "b": [k[5:] for k in COMP if test.loc[i, k] == -1],
             "p": round(float(test.loc[i, "p"]), 4), "gano": "A" if test.loc[i, "real"] == 1 else "B"}
            for i in dict.fromkeys(idx)]

n_cortes = len(CORTES)
por = int(por_corte.iloc[0]) if por_corte.nunique() == 1 else None
descartes = [("Algún equipo sin seis héroes", "incompleto", "Abandonos y datos parciales."),
             ("Héroe fuera del catálogo", "catalogo", f"Solo los {len(COMP)} héroes activos al momento de los cortes."),
             ("Un héroe repetido", "repetido", "Partidas con el mismo héroe en los dos equipos."),
             ("Sin ganador", "ganador", "Partidas sin un equipo ganador definido.")]
fila = lambda paso, n, que: f"<tr><td>{paso}</td><td class='n'>{n}</td><td>{que}</td></tr>"
tabla_filtros = ("<div class='tabla'><table><thead><tr><th>Paso</th><th>Partidas</th><th>Qué se hizo</th></tr></thead><tbody>"
                 + fila("Pedidas a la API", entero(embudo["leidas"]),
                        f"Ranked y en modo normal, desde cada fecha de corte y en orden de <code>match_id</code>: "
                        f"las primeras {entero(por) if por else 'N'} de cada fecha.")
                 + "".join(fila(paso, ("−" + entero(embudo[k])) if embudo[k] else "0", que) for paso, k, que in descartes)
                 + fila("En el dataset", entero(len(df)),
                        (f"{entero(por)} por fecha: " if por else "") + f"{NUMEROS.get(n_cortes - 1, n_cortes - 1)} para entrenar y validar, "
                        f"la del {fecha(CORTE_TEST)} para el test.")
                 + "</tbody></table></div>")
tabla_comparacion = ("<div class='tabla'><table><thead><tr><th>Modelo</th><th>Columnas</th><th>AUC (media ± desvío)</th></tr></thead><tbody>"
                     + "".join(f"<tr class='{'elegido' if n == ELEGIDO_NOMBRE else ''}'><td>{n}</td><td>{cols}</td>"
                               f"<td class='n'>{num(m, 4)} ± {num(s, 4)}</td></tr>" for n, (m, s, cols) in comparacion.items())
                     + "</tbody></table></div>")

e2_ok = NOTEBOOK_E2.exists() and all(v in NOTEBOOK_E2.read_text(encoding="utf-8") for k, v in ENTREGA_2.items() if k != "e2_corte")
if not e2_ok:
    avisos.append("los números de las hipótesis no aparecen en el notebook de la Entrega 2: revisar ENTREGA_2")
mid = dec["pct"].iloc[3:7]
VALORES = {
    "partidas": entero(len(df)), "test_n": entero(int(es_test.sum())), "por_corte": entero(por) if por else "unas " + entero(int(por_corte.mean())),
    "acc": pct(acc_test), "acc_frase": f"{round(100 * acc_test)} de cada 100", "piso": pct(piso_acc), "piso_clase": piso_clase,
    "top": num(quint["pct"].iloc[-1]) + " %", "top_frase": frase_corta(quint["pct"].iloc[-1]), "low": num(quint["pct"].iloc[0]) + " %",
    "r": num(r_api, 2), "auc": num(auc_test, 3), "fecha_test": fecha(CORTE_TEST), "fecha_train": FECHA_TRAIN,
    "n_cortes_txt": NUMEROS.get(n_cortes, str(n_cortes)), "fecha_ini": fecha(CORTES[0]), "fecha_fin": fecha(CORTES[-1]),
    "fecha_fin_larga": fecha(CORTES[-1], anio=True),
    "fechas_lista": ", ".join(fecha(c) for c in CORTES[:-1]) + f" y {fecha(CORTES[-1])} de {CORTES[-1][:4]}",
    "cmin": num(datos_cortes["pct"].min()) + " %", "cmax": num(datos_cortes["pct"].max()) + " %",
    "n_columnas": str(len(schema.COLUMNS)), "n_comp": str(len(COMP)),
    "n_ventajas": str(len(schema.COUNTER + schema.SYNERGY + schema.HERO_WINRATE)),
    "tabla_filtros": tabla_filtros, "tabla_comparacion": tabla_comparacion,
    "dec1": num(dec["pct"].iloc[0]) + " %", "dec10": num(dec["pct"].iloc[-1]) + " %",
    "dec1_frase": frase_fraccion(dec["pct"].iloc[0]), "dec10_frase": frase_fraccion(dec["pct"].iloc[-1]),
    "mmin": num(mid.min()) + " %", "mmax": num(mid.max()) + " %", "por_decil": entero(int(dec["size"].median())),
    **ENTREGA_2,
    "pasos": str(cv.get_n_splits(groups=g_train)), "azar": num(abs(azar), 3),
    "c_elegido": num(C_ELEGIDO, 1), "c_min": num(min(por_C), 4), "c_max": num(max(por_C), 4), "auc_e3": num(auc_e3, 3),
    "heroes_mas": f"{H.loc[mas.index[0], 'nombre']} ({num(mas.iloc[0])} puntos para su equipo) y {H.loc[mas.index[1], 'nombre']}".replace("(", "(+", 1),
    "heroes_menos": f"{H.loc[menos.index[0], 'nombre']} ({num(menos.iloc[0])}) y {H.loc[menos.index[1], 'nombre']}",
    "b0": num(b0, 3), "n_cruces": entero(len(COMP) * (len(COMP) - 1) // 2), "auc_cv_3": num(auc_cv, 3), "auc_cruces": num(auc_cruces, 3),
    "generada": dt.date.today().strftime("%d/%m/%Y"),
}
DATOS = {"b0": b0, "heroes": heroes, "ejemplos": ejemplos, "fecha_test": fecha(CORTE_TEST), "fecha_train": FECHA_TRAIN}

html = PLANTILLA.read_text(encoding="utf-8")
for k, v in VALORES.items():
    html = html.replace("{{" + k + "}}", v)
faltan = sorted(set(part.split("}}")[0] for part in html.split("{{")[1:]))
assert not faltan, f"la plantilla tiene marcadores sin valor: {faltan}"
html = (html.replace("/*__DATOS__*/null", json.dumps(DATOS, ensure_ascii=False))
            .replace("/*__SPECS__*/null", json.dumps(SPECS, ensure_ascii=False)))

SALIDA.mkdir(exist_ok=True)
for nombre, url in VENDOR.items():
    descargar(url, SALIDA / "vendor" / nombre)
(SALIDA / ".nojekyll").write_text("", encoding="utf-8")
(SALIDA / "index.html").write_text(html, encoding="utf-8")

# =====================================================================================
# Resumen
# =====================================================================================
print(f"\nescrita {(SALIDA / 'index.html').relative_to(RAIZ)} ({len(html.encode()) / 1024:.0f} KB)")
print(f"partidas: {len(df)} en {n_cortes} cortes · test {CORTE_TEST} · entrenado con {', '.join(ultimos)}")
print(f"embudo: {dict(embudo)}")
print(f"test: AUC {auc_test:.4f} · accuracy {acc_test:.4f} · piso {piso_acc:.4f} · modelo de la Entrega 3 {auc_e3:.4f}")
print("walk-forward:")
for n, (m, s, _) in comparacion.items():
    print(f"   {m:.4f} ± {s:.4f}  {n}")
print(f"al azar {azar:+.4f} · C 0,01 a 10: {min(por_C):.4f}–{max(por_C):.4f} · con cruces {auc_cruces:.4f} · r con la API {r_api:.3f}")
print("coincide con el notebook:", "sí" if coincide is not None and coincide <= 1e-9 else f"NO ({coincide})")
for a in avisos:
    print("AVISO:", a)
