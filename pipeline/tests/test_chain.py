from types import SimpleNamespace

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import pipeline.chain as chain_module
from pipeline.chain import RespuestaIncompletaError, _validar_salida, build_chain, process_text
from pipeline.schemas import EntidadTecnica, NivelCriticidad

ENTIDAD_VALIDA = EntidadTecnica(
    tecnologias=["FastAPI", "Redis"],
    nivel_de_criticidad=NivelCriticidad.ALTA,
    resumen_tecnico="Cuello de botella en el pool de conexiones.",
)


def _raw(finish_reason=None, stop_reason=None):
    metadata = {}
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    if stop_reason is not None:
        metadata["stop_reason"] = stop_reason
    return SimpleNamespace(response_metadata=metadata)


def resultado_ok(entidad):
    return {"raw": _raw(finish_reason="stop"), "parsed": entidad, "parsing_error": None}


def resultado_truncado(finish_reason="max_tokens"):
    return {"raw": _raw(finish_reason=finish_reason), "parsed": None, "parsing_error": ValueError("json incompleto")}


def resultado_parsing_fallido():
    return {"raw": _raw(finish_reason="stop"), "parsed": None, "parsing_error": ValueError("json mal formado")}


class FakeStructuredModel:
    """Doble de un chat model ya bindeado a `.with_structured_output()`: la
    'llamada a la API' ocurre acá (cuenta invocaciones, devuelve resultados
    en cola) en vez de pegarle a la red real."""

    def __init__(self, resultados):
        self._resultados = list(resultados)
        self.calls = 0

    def with_structured_output(self, schema, include_raw=False):
        async def _invocar(entrada):
            self.calls += 1
            resultado = self._resultados[self.calls - 1]
            if isinstance(resultado, Exception):
                raise resultado
            return resultado

        return RunnableLambda(_invocar)


# --- _validar_salida: lógica de resiliencia en aislamiento, sin red ---


def test_validar_salida_devuelve_el_objeto_parseado_cuando_todo_ok():
    assert _validar_salida(resultado_ok(ENTIDAD_VALIDA)) is ENTIDAD_VALIDA


@pytest.mark.parametrize("clave,valor", [
    ("finish_reason", "length"),      # OpenAI
    ("finish_reason", "max_tokens"),
    ("stop_reason", "max_tokens"),    # Anthropic
])
def test_validar_salida_rechaza_respuesta_cortada_por_limite_de_tokens(clave, valor):
    resultado = {"raw": _raw(**{clave: valor}), "parsed": None, "parsing_error": None}
    with pytest.raises(RespuestaIncompletaError):
        _validar_salida(resultado)


def test_validar_salida_rechaza_cuando_el_parseo_a_pydantic_fallo():
    with pytest.raises(RespuestaIncompletaError):
        _validar_salida(resultado_parsing_fallido())


def test_validar_salida_rechaza_si_no_hay_objeto_parseado_aunque_no_haya_parsing_error():
    resultado = {"raw": _raw(finish_reason="stop"), "parsed": None, "parsing_error": None}
    with pytest.raises(RespuestaIncompletaError):
        _validar_salida(resultado)


# --- _build_model: selección de proveedor (sin red, construir un cliente no llama a la API) ---


def test_build_model_selecciona_la_clase_correcta_por_proveedor():
    assert isinstance(chain_module._build_model("openai"), ChatOpenAI)
    assert isinstance(chain_module._build_model("anthropic"), ChatAnthropic)
    assert isinstance(chain_module._build_model("ollama"), ChatOllama)


def test_build_model_rechaza_proveedor_desconocido():
    with pytest.raises(ValueError):
        chain_module._build_model("mistral")


def test_build_model_respeta_max_tokens_override():
    modelo = chain_module._build_model("anthropic", max_tokens=15)
    assert modelo.max_tokens == 15


def test_build_model_ollama_usa_num_predict_en_vez_de_max_tokens():
    modelo = chain_module._build_model("ollama", max_tokens=15)
    assert modelo.num_predict == 15


def test_build_model_ollama_saca_el_sufijo_v1_de_ollama_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    modelo = chain_module._build_model("ollama")
    assert modelo.base_url == "http://localhost:11434"


# --- build_chain / process_text: retry end-to-end con el modelo mockeado ---


async def test_build_chain_recupera_tras_dos_fallos_en_el_tercer_intento(monkeypatch):
    """Caso crítico (igual que en el Módulo 1): falla 2 veces por respuesta
    truncada y se recupera en el 3er intento. El modelo debe haberse llamado
    exactamente 3 veces."""
    fake_model = FakeStructuredModel([resultado_truncado(), resultado_truncado(), resultado_ok(ENTIDAD_VALIDA)])
    monkeypatch.setattr(chain_module, "_build_model", lambda *a, **k: fake_model)

    chain = build_chain(provider="anthropic")
    resultado = await chain.ainvoke({"texto": "cualquier texto"})

    assert resultado is ENTIDAD_VALIDA
    assert fake_model.calls == 3


async def test_build_chain_se_rinde_tras_agotar_los_reintentos(monkeypatch):
    """Caso crítico: falla siempre. Se agota `MAX_RETRY_ATTEMPTS` sin
    recuperarse y la excepción sí se propaga (acá, a diferencia del Módulo 1,
    el diseño elegido es dejar que `RespuestaIncompletaError` suba, no
    devolver un objeto de error estructurado)."""
    fake_model = FakeStructuredModel([resultado_truncado()] * chain_module.MAX_RETRY_ATTEMPTS)
    monkeypatch.setattr(chain_module, "_build_model", lambda *a, **k: fake_model)

    chain = build_chain(provider="anthropic")
    with pytest.raises(RespuestaIncompletaError):
        await chain.ainvoke({"texto": "cualquier texto"})

    assert fake_model.calls == chain_module.MAX_RETRY_ATTEMPTS


async def test_process_text_retorna_la_entidad_validada(monkeypatch):
    fake_model = FakeStructuredModel([resultado_ok(ENTIDAD_VALIDA)])
    monkeypatch.setattr(chain_module, "_build_model", lambda *a, **k: fake_model)

    resultado = await process_text("cualquier texto", provider="anthropic")

    assert resultado is ENTIDAD_VALIDA


async def test_process_text_propaga_la_excepcion_tras_agotar_reintentos(monkeypatch):
    fake_model = FakeStructuredModel([resultado_truncado()] * chain_module.MAX_RETRY_ATTEMPTS)
    monkeypatch.setattr(chain_module, "_build_model", lambda *a, **k: fake_model)

    with pytest.raises(RespuestaIncompletaError):
        await process_text("cualquier texto", provider="anthropic")
