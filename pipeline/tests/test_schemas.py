import pytest
from pydantic import ValidationError

from pipeline.schemas import EntidadTecnica, NivelCriticidad


def test_acepta_datos_validos():
    entidad = EntidadTecnica(
        tecnologias=["FastAPI", "Redis", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico="Cuello de botella en el pool de conexiones.",
    )
    assert entidad.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert entidad.nivel_de_criticidad == NivelCriticidad.ALTA
    assert entidad.resumen_tecnico == "Cuello de botella en el pool de conexiones."


def test_rechaza_lista_de_tecnologias_vacia():
    with pytest.raises(ValidationError):
        EntidadTecnica(tecnologias=[], nivel_de_criticidad="baja", resumen_tecnico="ok")


@pytest.mark.parametrize("nivel", ["baja", "media", "alta"])
def test_acepta_los_tres_niveles_de_criticidad(nivel):
    entidad = EntidadTecnica(tecnologias=["Redis"], nivel_de_criticidad=nivel, resumen_tecnico="ok")
    assert entidad.nivel_de_criticidad == nivel


def test_rechaza_nivel_de_criticidad_invalido():
    with pytest.raises(ValidationError):
        EntidadTecnica(tecnologias=["Redis"], nivel_de_criticidad="critica", resumen_tecnico="ok")


def test_rechaza_resumen_tecnico_vacio():
    with pytest.raises(ValidationError):
        EntidadTecnica(tecnologias=["Redis"], nivel_de_criticidad="baja", resumen_tecnico="")


@pytest.mark.parametrize("campo_faltante", ["tecnologias", "nivel_de_criticidad", "resumen_tecnico"])
def test_rechaza_campos_requeridos_faltantes(campo_faltante):
    data = {"tecnologias": ["Redis"], "nivel_de_criticidad": "baja", "resumen_tecnico": "ok"}
    del data[campo_faltante]
    with pytest.raises(ValidationError):
        EntidadTecnica(**data)


def test_rechaza_tecnologias_con_tipo_incorrecto():
    with pytest.raises(ValidationError):
        EntidadTecnica(tecnologias="Redis", nivel_de_criticidad="baja", resumen_tecnico="ok")
