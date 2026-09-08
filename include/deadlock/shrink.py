"""
Shrinkage bayesiano (regresion a la media) - Problema 4 de
docs/decisiones-diseno-dataset.md.

Un winrate calculado sobre 20 partidas y otro sobre 5.000 no tienen la misma
confiabilidad, pero sin esto el modelo los recibiria como el mismo numero.
La formula empuja las tasas con poca muestra hacia un valor neutro, y deja
casi intactas las que tienen mucha historia detras -- la misma logica que
usa IMDb para rankear peliculas con pocas resenas.

k=100 se eligio con el criterio estadistico documentado en el Problema 4:
el error estandar de una proporcion (0.5/sqrt(n)) baja a ~5 puntos
porcentuales recien alrededor de n=100. Es una constante de pipeline, no un
numero fijo de por vida: en la Unidad 3 se puede recalcular tratandolo como
hiperparametro y comparando contra un conjunto de validacion.
"""

K_DEFAULT = 100


def shrink_rate(wins: float, matches: float, k: int = K_DEFAULT) -> float:
    """(wins + k*0.5) / (matches + k), y lo centra en 0: 2*eso - 1.

    matches=0 (pareja ausente de la respuesta de la API) cae exactamente en
    el valor neutro (0.0 centrado) sin necesidad de una regla de imputacion
    aparte -- es el caso limite de la misma formula.
    """
    winrate = (wins + k * 0.5) / (matches + k)
    return 2 * winrate - 1


def shrink_diff(diff: float, matches: float, k: int = K_DEFAULT) -> float:
    """Encoge una diferencia (ya centrada en 0) hacia 0 segun la muestra."""
    if matches + k == 0:
        return 0.0
    return diff * (matches / (matches + k))
